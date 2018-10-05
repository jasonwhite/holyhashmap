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

    pub fn zero() -> Self {
        HashValue(0)
    }

    /// Returns the index into which this hash value should go given an array
    /// length.
    pub fn index(&self, mask: usize) -> usize {
        debug_assert!((mask + 1).is_power_of_two());
        (self.0 & mask as u64) as usize
    }
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
        let capacity = if capacity == 0 {
            0
        } else {
            // Always use a power-of-two capacity so that we can use `&` instead
            // of `%` for determining the bucket.
            //
            // Double the user-supplied capacity to maintain the max load
            // factor.
            max(32, (capacity * 2).next_power_of_two())
        };

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

    #[inline]
    pub fn get_index<Q>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.get_index(key)
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
        self.inner.get(key)
    }

    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.insert_index(key, value).1
    }

    #[inline]
    pub fn insert_index(&mut self, key: K, value: V) -> (usize, Option<V>) {
        let hash = HashValue::new(&self.hash_builder, &key);
        self.inner.insert_index(hash, key, value)
    }

    #[inline]
    pub fn remove_index(&mut self, index: usize) -> (K, V) {
        self.inner.remove_index(index)
    }

    #[inline]
    pub fn remove<Q>(&mut self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.remove_entry(k).map(|(_, v)| v)
    }

    pub fn remove_entry<Q>(&mut self, k: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match self.inner.get_index(k) {
            Some(index) => Some(self.remove_index(index)),
            None => None,
        }
    }

    #[inline]
    pub fn iter(&self) -> Iter<K, V> {
        self.inner.iter()
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
        let _reserve = if self.is_empty() {
            iter.size_hint().0
        } else {
            (iter.size_hint().0 + 1) / 2
        };

        // self.reserve(reserve);

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

const BUCKET_EMPTY: usize = usize::max_value();
const BUCKET_TOMBSTONE: usize = usize::max_value() - 1;

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
    pub const TOMBSTONE: Bucket = Bucket {
        hash: HashValue(0),
        index: BUCKET_TOMBSTONE,
    };

    /// Creates a bucket with the given index and hash value.
    pub fn new(hash: HashValue, index: usize) -> Self {
        Bucket { hash, index }
    }

    pub fn is_tombstone(&self) -> bool {
        self == &Self::EMPTY
    }

    pub fn is_empty(&self) -> bool {
        self == &Self::TOMBSTONE
    }
}

/// A tombstone.
#[derive(Debug, Clone, Eq, PartialEq)]
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

#[derive(Clone)]
struct InnerMap<K, V> {
    // The buckets. This is the vector we do linear probing on. When the
    // correct bucket is found, we use it to index into the vector of
    // entries.
    //
    // The length of this vector must always be a power of 2 such that we can
    // use the &-operator to index into it (AND is faster than MOD).
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
    pub fn with_capacity(n: usize) -> Self {
        InnerMap {
            buckets: vec![Bucket::EMPTY; n],
            entries: Vec::new(),
            tombstones: 0,
            last_tombstone: Tombstone::HEAD,
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

    // Double the capacity of the buckets. This filters out all bucket
    // tombstones.
    pub fn double_capacity(&mut self) {
        let new_capacity = self.capacity() * 2;

        let old_buckets =
            mem::replace(&mut self.buckets, vec![Bucket::EMPTY; new_capacity]);

        let mask = new_capacity.wrapping_sub(1);

        // For each old bucket, reinsert it into the new buckets at the right
        // spot.
        for bucket in old_buckets {
            if bucket.is_empty() || bucket.is_tombstone() {
                continue;
            }

            let mut index = bucket.hash.index(new_capacity);

            // Probe for an empty bucket and place it there.
            loop {
                let mut candidate = self.buckets.get_mut(index).unwrap();

                if candidate.is_empty() {
                    *candidate = bucket;
                    break;
                }

                // Wrap around to the beginning of the array if necessary.
                index = index.wrapping_add(1) & mask;
            }
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        let new_size = self.len() + additional;
        if new_size > self.capacity() {
            self.double_capacity()
        }
    }
}

impl<K, V> InnerMap<K, V>
where
    K: Hash + Eq,
{
    pub fn get_index<Q>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        None
    }

    /// Returns an index to a bucket where an entry has been found or can be
    /// inserted at.
    pub fn search<Q>(&self, hash: HashValue, key: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let mask = self.capacity().wrapping_sub(1);
        let mut index = hash.index(mask);

        loop {
            let bucket = self.buckets.get(index).unwrap();

            if bucket.is_empty() {
                return index;
            } else if bucket.is_tombstone() {
                // Skip over tombstones.
                continue;
            } else if bucket.hash == hash {
                // The hash matches. Make sure the key actually matches.
                match self.entries.get(bucket.index).unwrap() {
                    BucketEntry::Entry(k, _) => {
                        if k.borrow() == key {
                            return index;
                        }
                    }
                    _ => unreachable!(), // This should be impossible.
                }
            }

            // Wrap around to the beginning of the array if necessary.
            index = index.wrapping_add(1) & mask;
        }
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        None
    }

    pub fn insert_index(&mut self, hash: HashValue, key: K, value: V) -> (usize, Option<V>) {
        let mask = self.capacity().wrapping_sub(1);
        let index = hash.index(mask);

        (0, None)
    }

    pub fn remove_index(&mut self, index: usize) -> (K, V) {
        unimplemented!()
    }

    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            iter: self.entries.iter(),
        }
    }
}

pub struct Iter<'a, K: 'a, V: 'a> {
    iter: slice::Iter<'a, BucketEntry<K, V>>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.iter.next() {
            match entry {
                BucketEntry::Entry(k, v) => return Some((&k, &v)),
                BucketEntry::Tombstone(_) => {}
            }
        }

        None
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // Simplify the type name so that we can use the exact same tests as std's
    // HashMap.
    type HashMap<K, V> = HolyHashMap<K, V>;

    #[test]
    fn test_create_capacity_zero() {
        let mut m = HashMap::with_capacity(0);

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
