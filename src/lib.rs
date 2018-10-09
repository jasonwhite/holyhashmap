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
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::FromIterator;
use std::mem;
use std::ops::Index;
use std::slice;

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
fn second<K, V>(kv: (K, V)) -> V {
    kv.1
}

/// > Holy hash maps, Batman!
/// > -- <cite>Robin</cite>
#[derive(Clone)]
pub struct HolyHashMap<K, V, S = RandomState> {
    hash_builder: S,
    inner: InnerMap<K, V>,
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

    pub fn capacity(&self) -> usize {
        // Capacity when the max load factor is taken into account.
        self.inner.max_load()
    }

    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    pub fn shrink_to_fit(&mut self) {
        let new_capacity = self.len();
        if new_capacity == 0 {
            self.inner = InnerMap::with_capacity(0);
        } else {
            self.inner.resize(new_capacity);
        }
    }

    #[inline]
    pub fn get_index<Q>(&self, key: &Q) -> Option<usize>
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
            // Can't compute hash for empty map.
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
    pub fn insert_full(&mut self, key: K, value: V) -> (usize, Option<V>) {
        // Must reserve additional space before calculating the hash.
        self.reserve(1);

        let hash = HashValue::new(&self.hash_builder, &key);
        self.inner.insert_full(hash, key, value)
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
        self.inner.remove(hash, key)
    }

    #[inline]
    pub fn iter(&self) -> Iter<K, V> {
        self.inner.iter()
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
            tombstones: self.inner.tombstones,
        }
    }
}

const BUCKET_EMPTY: usize = usize::max_value();

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
    index: usize,
}

impl Bucket {
    pub const EMPTY: Bucket = Bucket {
        hash: HashValue(0),
        index: BUCKET_EMPTY,
    };

    /// Creates a bucket with the given index and hash value.
    pub fn new(hash: HashValue, index: usize) -> Self {
        Bucket { hash, index }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.index == BUCKET_EMPTY
    }

    #[inline]
    pub fn index(&self, mask: usize) -> usize {
        self.hash.index(mask)
    }
}

/// A tombstone.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct Tombstone(usize);

impl Tombstone {
    const HEAD: Tombstone = Tombstone(usize::max_value());

    pub fn new(prev: usize) -> Self {
        Tombstone(prev)
    }

    pub fn is_head(&self) -> bool {
        self == &Self::HEAD
    }
}

/// A bucket entry. `None` if the entry has been deleted (i.e., it is
/// a tombstone).
#[derive(Debug, Clone)]
enum BucketEntry<K, V> {
    Tombstone(Tombstone),
    Entry(K, V),
}

impl<K, V> BucketEntry<K, V> {
    pub fn into_entry(self) -> (K, V) {
        match self {
            BucketEntry::Entry(k, v) => (k, v),
            _ => unreachable!(),
        }
    }

    pub fn entry(&self) -> (&K, &V) {
        match self {
            BucketEntry::Entry(k, v) => (k, v),
            _ => unreachable!(),
        }
    }

    pub fn tombstone(&self) -> &Tombstone {
        match self {
            BucketEntry::Tombstone(t) => t,
            _ => unreachable!(),
        }
    }

    #[cfg(test)]
    pub fn is_tombstone(&self) -> bool {
        match self {
            BucketEntry::Tombstone(_) => true,
            BucketEntry::Entry(_, _) => false,
        }
    }
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
    entries: Vec<BucketEntry<K, V>>,

    // The number of tombstones in the table (i.e., the number of non-`None`
    // values in `entries`). This is useful to give an accurate length of the
    // table.
    tombstones: usize,

    // Index of the last tombstone. This is used to reclaim deleted entries
    // such that indices are not invalidated upon removals.
    last_tombstone: Tombstone,
}

impl<K, V> InnerMap<K, V> {
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            // Don't allocate if the desired capacity is 0.
            InnerMap {
                mask: 0,
                buckets: Vec::new(),
                entries: Vec::new(),
                tombstones: 0,
                last_tombstone: Tombstone::HEAD,
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
                tombstones: 0,
                last_tombstone: Tombstone::HEAD,
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len() - self.tombstones
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
}

impl<K, V> InnerMap<K, V>
where
    K: Hash + Eq,
{
    pub fn get_index<Q>(&self, hash: HashValue, key: &Q) -> Option<usize>
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
            Some(self.entries.get(bucket.index).unwrap().entry())
        }
    }

    /// Returns an index to a bucket where an entry has been found or can be
    /// inserted at.
    pub fn search<Q>(&self, hash: HashValue, key: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let mut index = hash.index(self.mask);

        loop {
            let bucket = self.buckets.get(index).unwrap();

            if bucket.is_empty() {
                return index;
            } else if bucket.hash == hash {
                // The hash matches. Make sure the key actually matches.
                let (k, _) = self.entries[bucket.index].entry();
                if k.borrow() == key {
                    return index;
                }
            }

            // Wrap around to the beginning of the array if necessary.
            index = index.wrapping_add(1) & self.mask;
        }
    }

    pub fn remove<Q>(&mut self, hash: HashValue, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // Find the bucket we want to remove.
        let index = self.search(hash, key);

        // Remove the bucket, leaving an empty bucket in its place.
        let removed =
            mem::replace(self.buckets.get_mut(index).unwrap(), Bucket::EMPTY);

        if removed.is_empty() {
            // Already deleted.
            None
        } else {
            // Found the entry. Replace it with a tombstone and update the last
            // tombstone while we're at it.
            let entry = mem::replace(
                self.entries.get_mut(removed.index).unwrap(),
                BucketEntry::Tombstone(mem::replace(
                    &mut self.last_tombstone,
                    Tombstone::new(removed.index),
                )),
            ).into_entry();

            self.tombstones += 1;

            // We found an entry. We need to keep probing to find the last
            // bucket in this cluster and swap it into the deleted slot. Care
            // must be taken to only swap a bucket if it belongs in the same
            // cluster (i.e., it's hash value index is <= `i`).
            let mut i = index;
            let mut j = i.wrapping_add(1) & self.mask;
            loop {
                if self.buckets[j].is_empty() {
                    // Found an empty bucket. We're done.
                    break;
                } else if self.buckets[j].index(self.mask) <= i {
                    // Found a bucket that belongs in the same "group".
                    self.buckets.swap(i, j);

                    // The bucket at `i` is now empty. Continue the deletion
                    // process from here.
                    i = j;
                }

                j = j.wrapping_add(1) & self.mask;
            }

            Some(entry)
        }
    }

    /// Invariant: Space has already been reserved.
    pub fn insert_full(
        &mut self,
        hash: HashValue,
        key: K,
        value: V,
    ) -> (usize, Option<V>) {
        let i = self.search(hash, &key);
        let bucket = self.buckets.get_mut(i).unwrap();

        let entry = BucketEntry::Entry(key, value);

        if bucket.is_empty() {
            // Doesn't exist yet, we can go ahead and insert into this slot.
            let index = if self.last_tombstone.is_head() {
                // No tombstone to reuse. Just insert at the end.
                let index = self.entries.len();
                self.entries.push(entry);
                index
            } else {
                // Reuse a tombstone.
                let previous_tombstone =
                    *self.entries[self.last_tombstone.0].tombstone();

                let tombstone =
                    mem::replace(&mut self.last_tombstone, previous_tombstone);

                *self.entries.get_mut(tombstone.0).unwrap() = entry;

                self.tombstones -= 1;

                tombstone.0
            };

            *bucket = Bucket::new(hash, index);

            (index, None)
        } else {
            // Already exists. Update it with the new value.
            let index = bucket.index;

            let previous =
                mem::replace(self.entries.get_mut(index).unwrap(), entry)
                    .into_entry();

            (index, Some(previous.1))
        }
    }

    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            iter: self.entries.iter(),
            tombstones: self.tombstones,
        }
    }
}

#[derive(Clone)]
pub struct Iter<'a, K: 'a, V: 'a> {
    iter: slice::Iter<'a, BucketEntry<K, V>>,

    // Number of tombstones in the entries.
    tombstones: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.iter.next() {
            match entry {
                BucketEntry::Entry(k, v) => return Some((k, v)),
                BucketEntry::Tombstone(_) => {
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
            self.iter.nth(n).map(BucketEntry::entry)
        } else {
            let tombstones = &mut self.tombstones;
            self.iter
                .by_ref()
                .filter_map(move |entry| match entry {
                    BucketEntry::Entry(k, v) => return Some((k, v)),
                    BucketEntry::Tombstone(_) => {
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
                BucketEntry::Entry(k, v) => return Some((k, v)),
                BucketEntry::Tombstone(_) => {
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

#[derive(Clone)]
pub struct IntoIter<K, V> {
    iter: ::std::vec::IntoIter<BucketEntry<K, V>>,

    // Number of tombstones in the entries.
    tombstones: usize,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.iter.next() {
            match entry {
                BucketEntry::Entry(k, v) => return Some((k, v)),
                BucketEntry::Tombstone(_) => {
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
            self.iter.nth(n).map(BucketEntry::into_entry)
        } else {
            let tombstones = &mut self.tombstones;
            self.iter
                .by_ref()
                .filter_map(move |entry| match entry {
                    BucketEntry::Entry(k, v) => return Some((k, v)),
                    BucketEntry::Tombstone(_) => {
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
                BucketEntry::Entry(k, v) => return Some((k, v)),
                BucketEntry::Tombstone(_) => {
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

#[cfg(test)]
mod test {
    use super::*;
    use std::cell::RefCell;

    // Simplify the type name so that we can use the exact same tests as std's
    // HashMap.
    type HashMap<K, V> = HolyHashMap<K, V>;

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

    //#[test]
    // fn test_empty_iter() {
    //    let mut m: HashMap<i32, bool> = HashMap::new();
    //    assert_eq!(m.drain().next(), None);
    //    assert_eq!(m.keys().next(), None);
    //    assert_eq!(m.values().next(), None);
    //    assert_eq!(m.values_mut().next(), None);
    //    assert_eq!(m.iter().next(), None);
    //    assert_eq!(m.iter_mut().next(), None);
    //    assert_eq!(m.len(), 0);
    //    assert!(m.is_empty());
    //    assert_eq!(m.into_iter().next(), None);
    // }

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
                    assert!(
                        m.contains_key(&j),
                        format!("Key should exist: {}", j)
                    );
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
    fn insert() {
        let mut m = HashMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        assert_eq!(m.insert(2, 5), Some(4));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn remove() {
        let mut m = HashMap::new();
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        assert_eq!(m.remove(&2), Some(4));
        assert_eq!(m.len(), 1);
        assert_eq!(m.remove(&2), None);
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        assert_eq!(m.remove_entry(&2), Some((2, 4)));
    }
}
