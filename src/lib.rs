// Copyright (c) 2018 Jason White
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
use std::borrow::Borrow;
use std::cmp::max;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::{Enumerate, FromIterator, FusedIterator};
use std::mem;
use std::ops::Index;
use std::slice;

#[cfg(feature = "serde")]
pub mod serde;

/// > Holy hash maps, Batman!
/// > -- <cite>Robin</cite>
#[derive(Clone)]
pub struct HolyHashMap<K, V, S = RandomState> {
    hash_builder: S,
    inner: InnerMap<K, V>,
}

#[derive(Clone)]
struct InnerMap<K, V> {
    // The value to `&` with in order to derive the index of a bucket from
    // a hash value. This is always `capacity - 1` and since capacity is always
    // a power of two, the mask will be something like `0b1111`. It is faster
    // to use a bitwise AND than modulus to calculate the index.
    //
    // Note that it's not strictly necessary to store this here as it can
    // always be derived, but it is very convenient and avoids the need to
    // recalculate it for every table lookup.
    mask: usize,

    // The buckets. This is the vector we do linear probing on. When the
    // correct bucket is found, we use it to index into the vector of
    // entries.
    //
    // The length of this vector must always be a power of 2 such that we can
    // use the &-operator to index into it (see `mask` above).
    buckets: Vec<Bucket>,

    // The actual data. Key-value pairs are pushed onto the end of the vector.
    // The entires are guaranteed to not change position in the vector upon
    // insertion or deletion.
    entries: Vec<Option<(K, V)>>,

    // Indices of tombstones in the entries.
    tombstones: Vec<EntryIndex>,
}

/// A type-safe position in the entries vector. This type is intentionally
/// lacking trait implementions such as `PartialOrd` or `Add` in order to avoid
/// creating an invalid index.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct EntryIndex(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct HashValue(u64);

impl HashValue {
    pub fn new<T, S>(state: &S, t: &T) -> Self
    where
        T: Hash + ?Sized,
        S: BuildHasher,
    {
        let mut hasher = state.build_hasher();
        t.hash(&mut hasher);
        HashValue(hasher.finish())
    }

    /// Returns the index into which this hash value should go given an array
    /// length.
    pub fn index(&self, mask: usize) -> usize {
        debug_assert!(
            mask.wrapping_add(1).is_power_of_two(),
            format!("invalid mask {:x?}", mask)
        );
        (self.0 & mask as u64) as usize
    }
}

// Helper function for getting the second element in a tuple without generating
// a lambda function.
fn first<K, V>(kv: (K, V)) -> K {
    kv.0
}

// Helper function for getting the second element in a tuple without generating
// a lambda function.
fn second<K, V>(kv: (K, V)) -> V {
    kv.1
}

impl<K, V> HolyHashMap<K, V>
where
    K: Eq + Hash,
{
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> HolyHashMap<K, V, RandomState> {
        Self::with_capacity_and_hasher(capacity, Default::default())
    }
}

impl<K, V, S> HolyHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn with_hasher(hash_builder: S) -> Self {
        Self::with_capacity_and_hasher(0, hash_builder)
    }

    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        HolyHashMap {
            hash_builder,
            inner: InnerMap::with_capacity(capacity),
        }
    }

    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Maxium size before the map needs to reallocate. This is not the true
    /// capacity, but rather the max load of the map.
    pub fn capacity(&self) -> usize {
        // Capacity when the max load factor is taken into account.
        self.inner.max_load()
    }

    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);

        #[cfg(test)]
        self.check_consistency();
    }

    pub fn shrink_to_fit(&mut self) {
        let new_capacity = self.len();
        if new_capacity == 0 {
            self.inner = InnerMap::with_capacity(0);
        } else {
            self.inner.resize(new_capacity);
        }

        #[cfg(test)]
        self.check_consistency();
    }

    /// Gets the entry index of the key.
    pub fn key_index<Q>(&self, key: &Q) -> Option<EntryIndex>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.is_empty() {
            // Can't compute hash for empty map.
            return None;
        }

        let hash = HashValue::new(&self.hash_builder, &key);
        self.inner.get_index(hash, key)
    }

    /// Gets the entry by index.
    ///
    /// Returns `None` if the index is invalid (i.e., it is out of bounds or
    /// points to a deleted entry).
    #[inline]
    pub fn index_entry(&self, index: EntryIndex) -> Option<(&K, &V)> {
        self.inner.index_entry(index)
    }

    /// Gets the mutable entry by index.
    ///
    /// Returns `None` if the index is invalid (i.e., it is out of bounds or
    /// points to a deleted entry).
    #[inline]
    pub fn index_entry_mut(
        &mut self,
        index: EntryIndex,
    ) -> Option<(&K, &mut V)> {
        self.inner.index_entry_mut(index)
    }

    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.is_empty() {
            // Can't compute hash index for empty map.
            return None;
        }

        let hash = HashValue::new(&self.hash_builder, &key);
        self.inner.get_entry(hash, key).map(second)
    }

    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.insert_full(key, value).1
    }

    #[inline]
    pub fn insert_full(&mut self, key: K, value: V) -> (EntryIndex, Option<V>) {
        // Must reserve additional space before calculating the hash.
        self.reserve(1);

        let hash = HashValue::new(&self.hash_builder, &key);
        let result = self.inner.insert_full(hash, key, value);

        #[cfg(test)]
        self.check_consistency();

        result
    }

    #[inline]
    pub fn remove<Q>(&mut self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.remove_entry(k).map(second)
    }

    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.is_empty() {
            // Can't compute hash for empty map.
            return None;
        }

        let hash = HashValue::new(&self.hash_builder, &key);
        let result = self.inner.remove(hash, key);

        #[cfg(test)]
        self.check_consistency();

        result
    }

    #[inline]
    pub fn iter(&self) -> Iter<K, V> {
        self.inner.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        self.inner.iter_mut()
    }

    #[inline]
    pub fn keys(&self) -> Keys<K, V> {
        self.inner.keys()
    }

    #[inline]
    pub fn values(&self) -> Values<K, V> {
        self.inner.values()
    }

    #[inline]
    pub fn indices(&self) -> Indices<K, V> {
        self.inner.indices()
    }

    #[inline]
    pub fn values_mut(&mut self) -> ValuesMut<K, V> {
        self.inner.values_mut()
    }

    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<K, V> {
        self.reserve(1);

        let hash = HashValue::new(&self.hash_builder, &key);
        self.inner.entry(hash, key)
    }

    // Checks the internal consistency of the map. Helpful for finding bugs.
    #[cfg(test)]
    fn check_consistency(&self) {
        let capacity = self.inner.capacity();
        assert!(capacity == 0 || capacity.is_power_of_two());

        // There must be no index that points to a tombstone entry.
        for bucket in &self.inner.buckets {
            if !bucket.is_empty() {
                assert!(match self.inner.entries[bucket.index.0] {
                    None => false,
                    Some(_) => true,
                });
            }
        }

        // Check that the keys we have inserted actually exist in the buckets.
        for (i, entry) in self.inner.entries.iter().enumerate() {
            match entry {
                None => {}
                Some((k, _)) => {
                    // Check that the key exists and points to this index.
                    assert_eq!(self.key_index(k), Some(EntryIndex(i)));
                }
            }
        }
    }
}

impl<K, V, S> Default for HolyHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
{
    fn default() -> Self {
        Self::with_hasher(Default::default())
    }
}

impl<K, V, S> FromIterator<(K, V)> for HolyHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let mut map = HolyHashMap::with_hasher(Default::default());
        map.extend(iter);
        map
    }
}

impl<K, V, S> Extend<(K, V)> for HolyHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let iter = iter.into_iter();
        let reserve = if self.is_empty() {
            iter.size_hint().0
        } else {
            (iter.size_hint().0 + 1) / 2
        };

        self.reserve(reserve);

        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<'a, K, V, S> Extend<(&'a K, &'a V)> for HolyHashMap<K, V, S>
where
    K: Eq + Hash + Copy,
    V: Copy,
    S: BuildHasher,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (&'a K, &'a V)>,
    {
        self.extend(iter.into_iter().map(|(&key, &value)| (key, value)))
    }
}

impl<'a, K, Q, V, S> Index<&'a Q> for HolyHashMap<K, V, S>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash + ?Sized,
    S: BuildHasher,
{
    type Output = V;

    #[inline]
    fn index(&self, key: &Q) -> &Self::Output {
        self.get(key).expect("no entry found for key")
    }
}

impl<'a, K, V, S> IntoIterator for &'a HolyHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K, V, S> IntoIterator for HolyHashMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> IntoIter<K, V> {
        IntoIter {
            iter: self.inner.entries.into_iter(),
            tombstones: self.inner.tombstones.len(),
        }
    }
}

impl<K, V, S> PartialEq for HolyHashMap<K, V, S>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter()
            .all(|(key, value)| other.get(key).map_or(false, |v| *value == *v))
    }
}

impl<K, V, S> Eq for HolyHashMap<K, V, S>
where
    K: Eq + Hash,
    V: Eq,
    S: BuildHasher,
{}

impl<K, V, S> fmt::Debug for HolyHashMap<K, V, S>
where
    K: Eq + Hash + fmt::Debug,
    V: fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

const BUCKET_EMPTY: EntryIndex = EntryIndex(usize::max_value());

/// A bucket in the hash map. This doesn't actually store the key-value pair, it
/// points a vector that contains the key-value pair. This indirection helps
/// with cache locality and enables stable indices.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Bucket {
    /// The hash of the key. While not strictly necessary to store this here,
    /// we can use this to simplify rehashing and to make faster key
    /// comparisons. If the hash value matches, then we verify equality by
    /// doing a comparison on the real key.
    hash: HashValue,

    /// The index into the vector of entries.
    index: EntryIndex,
}

impl Bucket {
    /// Placeholder for an empty bucket entry.
    pub const EMPTY: Bucket = Bucket {
        hash: HashValue(0),
        index: BUCKET_EMPTY,
    };

    /// Creates a bucket with the given index and hash value.
    pub fn new(hash: HashValue, index: EntryIndex) -> Self {
        Bucket { hash, index }
    }

    /// `true` if this bucket is empty (i.e., it does not have a valid index to
    /// an entry).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.index == BUCKET_EMPTY
    }

    /// Derives the index of this bucket from the hash value using the map mask
    /// (i.e., the map capacity - 1).
    #[inline]
    pub fn index(&self, mask: usize) -> usize {
        self.hash.index(mask)
    }
}

impl<K, V> InnerMap<K, V> {
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            // Don't allocate if the desired capacity is 0.
            InnerMap {
                mask: 0,
                buckets: Vec::new(),
                entries: Vec::new(),
                tombstones: Vec::new(),
            }
        } else {
            // Always use a power-of-two capacity so that we can use `&` instead
            // of `%` for determining the bucket.
            //
            // Double the user-supplied capacity to maintain the max load
            // factor.
            let n = max(32, (capacity * 2).next_power_of_two());

            InnerMap {
                mask: n.wrapping_sub(1),
                buckets: vec![Bucket::EMPTY; n],
                entries: Vec::new(),
                tombstones: Vec::new(),
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len() - self.tombstones.len()
    }

    pub fn capacity(&self) -> usize {
        self.buckets.len()
    }

    /// The number of entries that can be stored without resizing.
    pub fn max_load(&self) -> usize {
        // Max load factor of 50%.
        self.capacity() / 2
    }

    pub fn reserve(&mut self, additional: usize) {
        let new_size = self.len() + additional;
        if new_size > self.max_load() {
            // Grow in terms of physical capacity, not max load.
            //
            // This might be our first allocation. Make sure we're not just
            // multiplying 0 by 2.
            self.resize(new_size);
        }
    }

    // Resize the map.
    pub fn resize(&mut self, capacity: usize) {
        let capacity = max(32, (capacity * 2).next_power_of_two());

        let old_buckets =
            mem::replace(&mut self.buckets, vec![Bucket::EMPTY; capacity]);

        self.mask = capacity.wrapping_sub(1);

        // For each old bucket, reinsert it into the new buckets at the right
        // spot.
        for bucket in old_buckets {
            if bucket.is_empty() {
                continue;
            }

            let mut index = bucket.index(self.mask);

            // Probe for an empty bucket and place it there.
            loop {
                let mut candidate = self.buckets.get_mut(index).unwrap();

                if candidate.is_empty() {
                    *candidate = bucket;
                    break;
                }

                // Wrap around to the beginning of the array if necessary.
                index = index.wrapping_add(1) & self.mask;
            }
        }
    }

    pub fn index_entry(&self, index: EntryIndex) -> Option<(&K, &V)> {
        self.entries
            .get(index.0)
            .and_then(|e| e.as_ref().map(|(k, v)| (k, v)))
    }

    pub fn index_entry_mut(
        &mut self,
        index: EntryIndex,
    ) -> Option<(&K, &mut V)> {
        self.entries
            .get_mut(index.0)
            .and_then(|e| e.as_mut().map(|(k, v)| (&*k, v)))
    }

    fn raw_entry_mut(&mut self, index: EntryIndex) -> &mut Option<(K, V)> {
        self.entries.get_mut(index.0).unwrap()
    }

    fn set_raw_entry(&mut self, index: EntryIndex, entry: Option<(K, V)>) {
        *self.entries.get_mut(index.0).unwrap() = entry;
    }

    // Removes an existing bucket.
    //
    // Precondition: The bucket is not empty.
    pub fn remove_bucket(&mut self, index: usize) -> (K, V) {
        // Remove the bucket, leaving an empty bucket in its place.
        let removed =
            mem::replace(self.buckets.get_mut(index).unwrap(), Bucket::EMPTY);

        // Found the entry. Remove it and update the tombstone list.
        let entry = self.raw_entry_mut(removed.index).take().unwrap();
        self.tombstones.push(removed.index);

        // We found an entry. We need to keep probing to find the last
        // bucket in this cluster and swap it into the deleted slot. Care
        // must be taken to only swap a bucket if it belongs in the same
        // cluster (i.e., it's hash value index is <= `i`).
        let mut i = index;
        let mut j = i.wrapping_add(1) & self.mask;
        while !self.buckets[j].is_empty() {
            let k = self.buckets[j].index(self.mask);

            let invalid_position = if j > i {
                k <= i || k > j
            } else {
                k <= i && k > j
            };

            if invalid_position {
                self.buckets.swap(i, j);

                // The bucket at `j` is now empty. Continue the deletion
                // process from here.
                i = j;
            }

            j = j.wrapping_add(1) & self.mask;
        }

        entry
    }
}

impl<K, V> InnerMap<K, V>
where
    K: Hash + Eq,
{
    pub fn get_index<Q>(&self, hash: HashValue, key: &Q) -> Option<EntryIndex>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let bucket = self.buckets.get(self.search(hash, key)).unwrap();

        if bucket.is_empty() {
            None
        } else {
            Some(bucket.index)
        }
    }

    pub fn get_entry<Q>(&self, hash: HashValue, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let bucket = self.buckets.get(self.search(hash, key)).unwrap();

        if bucket.is_empty() {
            None
        } else {
            Some(self.index_entry(bucket.index).unwrap())
        }
    }

    /// Returns an index to a bucket where an entry has been found or can be
    /// inserted at.
    pub fn search<Q>(&self, hash: HashValue, key: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let mut i = hash.index(self.mask);

        loop {
            let bucket = self.buckets.get(i).unwrap();

            if bucket.is_empty() {
                return i;
            } else if bucket.hash == hash {
                // The hash matches. Make sure the key actually matches.
                let (k, _) = self.index_entry(bucket.index).unwrap();
                if k.borrow() == key {
                    return i;
                }
            }

            // Wrap around to the beginning of the array if necessary.
            i = i.wrapping_add(1) & self.mask;
        }
    }

    pub fn remove<Q>(&mut self, hash: HashValue, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.search(hash, key);

        if self.buckets[index].is_empty() {
            None
        } else {
            Some(self.remove_bucket(index))
        }
    }

    /// Invariant: Space has already been reserved.
    pub fn insert_full(
        &mut self,
        hash: HashValue,
        key: K,
        value: V,
    ) -> (EntryIndex, Option<V>) {
        let i = self.search(hash, &key);

        if self.buckets[i].is_empty() {
            (self.insert_new(i, hash, key, value), None)
        } else {
            let entry = Some((key, value));

            // Already exists. Update it with the new value.
            let index = self.buckets[i].index;

            let previous =
                mem::replace(self.raw_entry_mut(index), entry).unwrap();

            (index, Some(previous.1))
        }
    }

    // Precondition: `bucket` points to an empty bucket where this entry can be
    // inserted.
    pub fn insert_new(
        &mut self,
        bucket: usize,
        hash: HashValue,
        key: K,
        value: V,
    ) -> EntryIndex {
        let entry = Some((key, value));

        let index = if let Some(index) = self.tombstones.pop() {
            // Reuse a tombstone.
            self.set_raw_entry(index, entry);
            index
        } else {
            // No tombstone to reuse. Just insert at the end.
            let index = self.entries.len();
            self.entries.push(entry);
            EntryIndex(index)
        };

        self.buckets[bucket] = Bucket::new(hash, index);

        index
    }

    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            iter: self.entries.iter(),
            tombstones: self.tombstones.len(),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        IterMut {
            iter: self.entries.iter_mut(),
            tombstones: self.tombstones.len(),
        }
    }

    pub fn keys(&self) -> Keys<K, V> {
        Keys { iter: self.iter() }
    }

    pub fn values(&self) -> Values<K, V> {
        Values { iter: self.iter() }
    }

    pub fn values_mut(&mut self) -> ValuesMut<K, V> {
        ValuesMut {
            iter: self.iter_mut(),
        }
    }

    pub fn indices(&self) -> Indices<K, V> {
        Indices {
            iter: self.entries.iter().enumerate(),
            tombstones: self.tombstones.len(),
        }
    }

    pub fn entry(&mut self, hash: HashValue, key: K) -> Entry<K, V> {
        let index = self.search(hash, &key);

        if self.buckets[index].is_empty() {
            Entry::Vacant(VacantEntry {
                map: self,
                index,
                hash,
                key,
            })
        } else {
            Entry::Occupied(OccupiedEntry { map: self, index })
        }
    }
}

pub enum Entry<'a, K: 'a, V: 'a> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

pub struct OccupiedEntry<'a, K: 'a, V: 'a> {
    map: &'a mut InnerMap<K, V>,

    // The position of the bucket where this entry lives. (Not the position in
    // the list of entries.)
    index: usize,
}

pub struct VacantEntry<'a, K: 'a, V: 'a> {
    map: &'a mut InnerMap<K, V>,

    // The position of the bucket where we can insert this entry.
    index: usize,

    // The hash of the key to insert.
    hash: HashValue,

    // The key to insert.
    key: K,
}

impl<'a, K, V> Entry<'a, K, V> {
    pub fn or_insert(self, default: V) -> &'a mut V
    where
        K: Eq + Hash,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default),
        }
    }

    pub fn or_insert_with<F>(self, default: F) -> &'a mut V
    where
        K: Eq + Hash,
        F: FnOnce() -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => entry.key(),
            Entry::Vacant(entry) => entry.key(),
        }
    }

    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        match self {
            Entry::Occupied(mut entry) => {
                f(entry.get_mut());
                Entry::Occupied(entry)
            }
            vacant => vacant,
        }
    }

    pub fn or_default(self) -> &'a mut V
    where
        K: Eq + Hash,
        V: Default,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(Default::default()),
        }
    }
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    pub fn index(&self) -> EntryIndex {
        self.map.buckets[self.index].index
    }

    pub fn key(&self) -> &K {
        self.map.index_entry(self.index()).unwrap().0
    }

    pub fn remove_entry(self) -> (K, V) {
        self.map.remove_bucket(self.index)
    }

    pub fn get(&self) -> &V {
        self.map.index_entry(self.index()).unwrap().1
    }

    pub fn get_mut(&mut self) -> &mut V {
        let entry_index = self.index();
        self.map.index_entry_mut(entry_index).unwrap().1
    }

    pub fn into_mut(self) -> &'a mut V {
        let entry_index = self.index();
        self.map.index_entry_mut(entry_index).unwrap().1
    }

    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    pub fn remove(self) -> V {
        self.map.remove_bucket(self.index).1
    }
}

impl<'a, K, V> VacantEntry<'a, K, V> {
    pub fn key(&self) -> &K {
        &self.key
    }

    pub fn into_key(self) -> K {
        self.key
    }

    pub fn index(&self) -> EntryIndex {
        self.map.buckets[self.index].index
    }

    pub fn insert(self, value: V) -> &'a mut V
    where
        K: Hash + Eq,
    {
        let entry_index =
            self.map.insert_new(self.index, self.hash, self.key, value);
        self.map.index_entry_mut(entry_index).unwrap().1
    }
}

#[derive(Clone)]
pub struct Iter<'a, K: 'a, V: 'a> {
    iter: slice::Iter<'a, Option<(K, V)>>,

    // Number of tombstones in the entries.
    tombstones: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.iter.next() {
            match entry {
                Some((k, v)) => return Some((k, v)),
                None => {
                    self.tombstones -= 1;
                }
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.tombstones == 0 {
            self.iter.nth(n).map(|entry| {
                let (k, v) = entry.as_ref().unwrap();
                (k, v)
            })
        } else {
            let tombstones = &mut self.tombstones;
            self.iter
                .by_ref()
                .filter_map(move |entry| match entry {
                    Some((k, v)) => return Some((k, v)),
                    None => {
                        *tombstones -= 1;
                        None
                    }
                }).nth(n)
        }
    }
}

impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a V)> {
        while let Some(entry) = self.iter.next_back() {
            match entry {
                Some((k, v)) => return Some((k, v)),
                None => {
                    self.tombstones -= 1;
                }
            }
        }

        None
    }
}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {
    fn len(&self) -> usize {
        self.iter.len() - self.tombstones
    }
}

impl<'a, K, V> FusedIterator for Iter<'a, K, V> {}

#[derive(Clone)]
pub struct IntoIter<K, V> {
    iter: ::std::vec::IntoIter<Option<(K, V)>>,

    // Number of tombstones in the entries.
    tombstones: usize,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.iter.next() {
            match entry {
                Some(e) => return Some(e),
                None => {
                    self.tombstones -= 1;
                }
            }
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.tombstones == 0 {
            self.iter.nth(n).map(Option::unwrap)
        } else {
            let tombstones = &mut self.tombstones;
            self.iter
                .by_ref()
                .filter_map(move |entry| match entry {
                    Some(e) => return Some(e),
                    None => {
                        *tombstones -= 1;
                        None
                    }
                }).nth(n)
        }
    }
}

impl<K, V> DoubleEndedIterator for IntoIter<K, V> {
    fn next_back(&mut self) -> Option<(K, V)> {
        while let Some(entry) = self.iter.next_back() {
            match entry {
                Some(e) => return Some(e),
                None => {
                    self.tombstones -= 1;
                }
            }
        }

        None
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize {
        self.iter.len() - self.tombstones
    }
}

impl<K, V> FusedIterator for IntoIter<K, V> {}

pub struct IterMut<'a, K: 'a, V: 'a> {
    iter: slice::IterMut<'a, Option<(K, V)>>,

    // Number of tombstones in the entries.
    tombstones: usize,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.iter.next() {
            match entry {
                Some((k, v)) => return Some((k, v)),
                None => {
                    self.tombstones -= 1;
                }
            }
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.tombstones == 0 {
            self.iter.nth(n).map(|entry| {
                let (k, v) = entry.as_mut().unwrap();
                (&*k, v)
            })
        } else {
            let tombstones = &mut self.tombstones;
            self.iter
                .by_ref()
                .filter_map(move |entry| match entry {
                    Some((ref k, ref mut v)) => return Some((k, v)),
                    None => {
                        *tombstones -= 1;
                        None
                    }
                }).nth(n)
        }
    }
}

impl<'a, K, V> DoubleEndedIterator for IterMut<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a mut V)> {
        while let Some(entry) = self.iter.next_back() {
            match entry {
                Some((k, v)) => return Some((k, v)),
                None => {
                    self.tombstones -= 1;
                }
            }
        }

        None
    }
}

impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V> {
    fn len(&self) -> usize {
        self.iter.len() - self.tombstones
    }
}

impl<'a, K, V> FusedIterator for IterMut<'a, K, V> {}

#[derive(Clone)]
pub struct Keys<'a, K: 'a, V: 'a> {
    iter: Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(first)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n).map(first)
    }
}

impl<'a, K, V> DoubleEndedIterator for Keys<'a, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a K> {
        self.iter.next_back().map(first)
    }
}

impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, K, V> FusedIterator for Keys<'a, K, V> {}

#[derive(Clone)]
pub struct Values<'a, K: 'a, V: 'a> {
    iter: Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(second)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n).map(second)
    }
}

impl<'a, K, V> DoubleEndedIterator for Values<'a, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a V> {
        self.iter.next_back().map(second)
    }
}

impl<'a, K, V> ExactSizeIterator for Values<'a, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, K, V> FusedIterator for Values<'a, K, V> {}

pub struct ValuesMut<'a, K: 'a, V: 'a> {
    iter: IterMut<'a, K, V>,
}

impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(second)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n).map(second)
    }
}

impl<'a, K, V> DoubleEndedIterator for ValuesMut<'a, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut V> {
        self.iter.next_back().map(second)
    }
}

impl<'a, K, V> ExactSizeIterator for ValuesMut<'a, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, K, V> FusedIterator for ValuesMut<'a, K, V> {}

#[derive(Clone)]
pub struct Indices<'a, K: 'a, V: 'a> {
    iter: Enumerate<slice::Iter<'a, Option<(K, V)>>>,

    // Number of tombstones in the entries.
    tombstones: usize,
}

impl<'a, K, V> Iterator for Indices<'a, K, V> {
    type Item = EntryIndex;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((i, entry)) = self.iter.next() {
            match entry {
                Some(_) => return Some(EntryIndex(i)),
                None => {
                    self.tombstones -= 1;
                }
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.tombstones == 0 {
            Some(EntryIndex(n))
        } else {
            let tombstones = &mut self.tombstones;
            self.iter
                .by_ref()
                .filter_map(move |(i, entry)| match entry {
                    Some(_) => return Some(EntryIndex(i)),
                    None => {
                        *tombstones -= 1;
                        None
                    }
                }).nth(n)
        }
    }
}

impl<'a, K, V> DoubleEndedIterator for Indices<'a, K, V> {
    fn next_back(&mut self) -> Option<EntryIndex> {
        while let Some((i, entry)) = self.iter.next_back() {
            match entry {
                Some(_) => return Some(EntryIndex(i)),
                None => {
                    self.tombstones -= 1;
                }
            }
        }

        None
    }
}

impl<'a, K, V> ExactSizeIterator for Indices<'a, K, V> {
    fn len(&self) -> usize {
        self.iter.len() - self.tombstones
    }
}

impl<'a, K, V> FusedIterator for Indices<'a, K, V> {}

#[cfg(test)]
mod test {
    // Simplify the type name so that we can use the exact same tests as std's
    // HashMap.
    use super::Entry::{Occupied, Vacant};
    use super::EntryIndex;
    use super::HolyHashMap as HashMap;
    use super::RandomState;
    use std::cell::RefCell;

    extern crate rand;

    #[test]
    fn test_zero_capacities() {
        type HM = HashMap<i32, i32>;

        let m = HM::new();
        assert_eq!(m.capacity(), 0);

        let m = HM::default();
        assert_eq!(m.capacity(), 0);

        let m = HM::with_hasher(RandomState::new());
        assert_eq!(m.capacity(), 0);

        let m = HM::with_capacity(0);
        assert_eq!(m.capacity(), 0);

        let m = HM::with_capacity_and_hasher(0, RandomState::new());
        assert_eq!(m.capacity(), 0);

        let mut m = HM::new();
        m.insert(1, 1);
        m.insert(2, 2);
        m.remove(&1);
        m.remove(&2);
        m.shrink_to_fit();
        assert_eq!(m.capacity(), 0);

        let mut m = HM::new();
        m.reserve(0);
        assert_eq!(m.capacity(), 0);
    }

    #[test]
    fn test_create_capacity_zero() {
        let mut m = HashMap::with_capacity(0);

        assert_eq!(m.capacity(), 0);

        assert!(m.insert(1, 1).is_none());

        assert!(m.contains_key(&1));
        assert!(!m.contains_key(&0));
    }

    #[test]
    fn test_insert() {
        let mut m = HashMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert_eq!(*m.get(&2).unwrap(), 4);
    }

    #[test]
    fn test_clone() {
        let mut m = HashMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        let m2 = m.clone();
        assert_eq!(*m2.get(&1).unwrap(), 2);
        assert_eq!(*m2.get(&2).unwrap(), 4);
        assert_eq!(m2.len(), 2);
    }

    thread_local! {
        static DROP_VECTOR: RefCell<Vec<i32>> = RefCell::new(Vec::new());
    }

    #[derive(Hash, PartialEq, Eq)]
    struct Droppable {
        k: usize,
    }

    impl Droppable {
        fn new(k: usize) -> Droppable {
            DROP_VECTOR.with(|slot| {
                slot.borrow_mut()[k] += 1;
            });

            Droppable { k: k }
        }
    }

    impl Drop for Droppable {
        fn drop(&mut self) {
            DROP_VECTOR.with(|slot| {
                slot.borrow_mut()[self.k] -= 1;
            });
        }
    }

    impl Clone for Droppable {
        fn clone(&self) -> Droppable {
            Droppable::new(self.k)
        }
    }

    #[test]
    fn test_drops() {
        DROP_VECTOR.with(|slot| {
            *slot.borrow_mut() = vec![0; 200];
        });

        {
            let mut m = HashMap::new();

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 0);
                }
            });

            for i in 0..100 {
                let d1 = Droppable::new(i);
                let d2 = Droppable::new(i + 100);
                m.insert(d1, d2);
            }

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            for i in 0..50 {
                let k = Droppable::new(i);
                let v = m.remove(&k);

                assert!(v.is_some());

                DROP_VECTOR.with(|v| {
                    assert_eq!(v.borrow()[i], 1);
                    assert_eq!(v.borrow()[i + 100], 1);
                });
            }

            DROP_VECTOR.with(|v| {
                for i in 0..50 {
                    assert_eq!(v.borrow()[i], 0);
                    assert_eq!(v.borrow()[i + 100], 0);
                }

                for i in 50..100 {
                    assert_eq!(v.borrow()[i], 1);
                    assert_eq!(v.borrow()[i + 100], 1);
                }
            });
        }

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 0);
            }
        });
    }

    #[test]
    fn test_into_iter_drops() {
        DROP_VECTOR.with(|v| {
            *v.borrow_mut() = vec![0; 200];
        });

        let hm = {
            let mut hm = HashMap::new();

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 0);
                }
            });

            for i in 0..100 {
                let d1 = Droppable::new(i);
                let d2 = Droppable::new(i + 100);
                hm.insert(d1, d2);
            }

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            hm
        };

        // By the way, ensure that cloning doesn't screw up the dropping.
        drop(hm.clone());

        {
            let mut half = hm.into_iter().take(50);

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            for _ in half.by_ref() {}

            DROP_VECTOR.with(|v| {
                let nk = (0..100).filter(|&i| v.borrow()[i] == 1).count();

                let nv = (0..100).filter(|&i| v.borrow()[i + 100] == 1).count();

                assert_eq!(nk, 50);
                assert_eq!(nv, 50);
            });
        };

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 0);
            }
        });
    }

    #[test]
    fn test_empty_remove() {
        let mut m: HashMap<i32, bool> = HashMap::new();
        assert_eq!(m.remove(&0), None);
    }

    #[test]
    fn test_empty_entry() {
        let mut m: HashMap<i32, bool> = HashMap::new();
        match m.entry(0) {
            Occupied(_) => panic!(),
            Vacant(_) => {}
        }
        assert!(*m.entry(0).or_insert(true));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_empty_iter() {
        let mut m: HashMap<i32, bool> = HashMap::new();
        // assert_eq!(m.drain().next(), None);
        assert_eq!(m.keys().next(), None);
        assert_eq!(m.values().next(), None);
        assert_eq!(m.values_mut().next(), None);
        assert_eq!(m.iter().next(), None);
        assert_eq!(m.iter_mut().next(), None);
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
        assert_eq!(m.into_iter().next(), None);
    }

    #[test]
    fn test_lots_of_insertions() {
        let mut m = HashMap::new();

        // Try this a few times to make sure we never screw up the hashmap's
        // internal state.
        for _ in 0..10 {
            assert!(m.is_empty());

            for i in 1..1001 {
                assert!(m.insert(i, i).is_none());

                for j in 1..i + 1 {
                    let r = m.get(&j);
                    assert_eq!(r, Some(&j));
                }

                for j in i + 1..1001 {
                    let r = m.get(&j);
                    assert_eq!(r, None);
                }
            }

            assert_eq!(m.len(), 1000);

            for i in 1001..2001 {
                assert!(!m.contains_key(&i));
            }

            // remove forwards
            for i in 1..1001 {
                assert!(m.remove(&i).is_some());

                for j in 1..i + 1 {
                    assert!(!m.contains_key(&j));
                }

                for j in i + 1..1001 {
                    assert!(m.contains_key(&j));
                }
            }

            for i in 1..1001 {
                assert!(!m.contains_key(&i));
            }

            for i in 1..1001 {
                assert!(m.insert(i, i).is_none());
            }

            // remove backwards
            for i in (1..1001).rev() {
                assert!(m.remove(&i).is_some());

                for j in i..1001 {
                    assert!(!m.contains_key(&j));
                }

                for j in 1..i {
                    assert!(m.contains_key(&j));
                }
            }
        }
    }

    #[test]
    fn test_iterate() {
        let mut m = HashMap::with_capacity(4);
        for i in 0..32 {
            assert!(m.insert(i, i * 2).is_none());
        }
        assert_eq!(m.len(), 32);

        let mut observed: u32 = 0;

        for (k, v) in &m {
            assert_eq!(*v, *k * 2);
            observed |= 1 << *k;
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_find() {
        let mut m = HashMap::new();
        assert!(m.get(&1).is_none());
        m.insert(1, 2);
        match m.get(&1) {
            None => panic!(),
            Some(v) => assert_eq!(*v, 2),
        }
    }

    #[test]
    fn test_from_iter() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HashMap<_, _> = xs.iter().cloned().collect();

        for &(k, v) in &xs {
            assert_eq!(map.get(&k), Some(&v));
        }
    }

    #[test]
    fn test_size_hint() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HashMap<_, _> = xs.iter().cloned().collect();

        let mut iter = map.iter();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_iter_len() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HashMap<_, _> = xs.iter().cloned().collect();

        let mut iter = map.iter();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_index() {
        let mut map = HashMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        assert_eq!(map[&2], 1);
    }

    #[test]
    #[should_panic]
    fn test_index_nonexistent() {
        let mut map = HashMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        map[&4];
    }

    #[test]
    fn test_entry() {
        let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

        let mut map: HashMap<_, _> = xs.iter().cloned().collect();

        // Existing key (insert)
        match map.entry(1) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                assert_eq!(view.get(), &10);
                assert_eq!(view.insert(100), 10);
            }
        }
        assert_eq!(map.get(&1).unwrap(), &100);
        assert_eq!(map.len(), 6);

        // Existing key (update)
        match map.entry(2) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                let v = view.get_mut();
                let new_v = (*v) * 10;
                *v = new_v;
            }
        }
        assert_eq!(map.get(&2).unwrap(), &200);
        assert_eq!(map.len(), 6);

        // Existing key (take)
        match map.entry(3) {
            Vacant(_) => unreachable!(),
            Occupied(view) => {
                assert_eq!(view.remove(), 30);
            }
        }
        assert_eq!(map.get(&3), None);
        assert_eq!(map.len(), 5);

        // Inexistent key (insert)
        match map.entry(10) {
            Occupied(_) => unreachable!(),
            Vacant(view) => {
                assert_eq!(*view.insert(1000), 1000);
            }
        }
        assert_eq!(map.get(&10).unwrap(), &1000);
        assert_eq!(map.len(), 6);
    }

    #[cfg(lolwut)]
    #[test]
    fn test_entry_take_doesnt_corrupt() {
        #![allow(deprecated)] // rand
                              // Test for #19292
        fn check(m: &HashMap<i32, ()>) {
            for k in m.keys() {
                assert!(
                    m.contains_key(k),
                    "{} is in keys() but not in the map?",
                    k
                );
            }
        }

        let mut m = HashMap::new();
        let mut rng = rand::thread_rng();

        // Populate the map with some items.
        for _ in 0..50 {
            let x = rng.gen_range(-10, 10);
            m.insert(x, ());
        }

        for i in 0..1000 {
            let x = rng.gen_range(-10, 10);
            match m.entry(x) {
                Vacant(_) => {}
                Occupied(e) => {
                    println!("{}: remove {}", i, x);
                    e.remove();
                }
            }

            check(&m);
        }
    }

    #[test]
    fn test_extend_ref() {
        let mut a = HashMap::new();
        a.insert(1, "one");
        let mut b = HashMap::new();
        b.insert(2, "two");
        b.insert(3, "three");

        a.extend(&b);

        assert_eq!(a.len(), 3);
        assert_eq!(a[&1], "one");
        assert_eq!(a[&2], "two");
        assert_eq!(a[&3], "three");
    }

    #[test]
    fn test_capacity_not_less_than_len() {
        let mut a = HashMap::new();
        let mut item = 0;

        for _ in 0..116 {
            a.insert(item, 0);
            item += 1;
        }

        assert!(a.capacity() > a.len());

        let free = a.capacity() - a.len();
        for _ in 0..free {
            a.insert(item, 0);
            item += 1;
        }

        assert_eq!(a.len(), a.capacity());

        // Insert at capacity should cause allocation.
        a.insert(item, 0);
        assert!(a.capacity() > a.len());
    }

    #[test]
    fn test_occupied_entry_key() {
        let mut a = HashMap::new();
        let key = "hello there";
        let value = "value goes here";
        assert!(a.is_empty());
        a.insert(key.clone(), value.clone());
        assert_eq!(a.len(), 1);
        assert_eq!(a[key], value);

        match a.entry(key.clone()) {
            Vacant(_) => panic!(),
            Occupied(e) => assert_eq!(key, *e.key()),
        }
        assert_eq!(a.len(), 1);
        assert_eq!(a[key], value);
    }

    #[test]
    fn test_vacant_entry_key() {
        let mut a = HashMap::new();
        let key = "hello there";
        let value = "value goes here";

        assert!(a.is_empty());
        match a.entry(key.clone()) {
            Occupied(_) => panic!(),
            Vacant(e) => {
                assert_eq!(key, *e.key());
                e.insert(value.clone());
            }
        }
        assert_eq!(a.len(), 1);
        assert_eq!(a[key], value);
    }

    #[test]
    fn test_indices() {
        let mut m: HashMap<_, _> = [("a", 0), ("b", 1), ("c", 2), ("d", 3)]
            .iter()
            .cloned()
            .collect();

        assert!(m.indices().eq(vec![0, 1, 2, 3].into_iter().map(EntryIndex)));

        m.remove("b");

        assert!(m.indices().eq(vec![0, 2, 3].into_iter().map(EntryIndex)));

        m.remove("a");
        m.insert("e", 4);

        assert!(m.indices().eq(vec![0, 2, 3].into_iter().map(EntryIndex)));
    }
}
