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
use std::collections::hash_map::{HashMap, RandomState, Entry};
use std::hash::{BuildHasher, Hash};
use std::iter::FromIterator;
use std::ops::Index;
use std::mem;
use std::slice;
use std::iter::Enumerate;

/// > Holy hash maps, Batman!
/// > -- <cite>Robin</cite>
#[derive(Clone)]
pub struct HolyHashMap<K, V, S = RandomState> {
    map: HashMap<K, usize, S>,
    values: Vec<Bucket<V>>,
    tombstones: usize,
    last_tombstone: Option<usize>,
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
        self.map.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        HolyHashMap {
            map: HashMap::with_capacity_and_hasher(capacity, hash_builder),
            values: Vec::with_capacity(capacity),
            tombstones: 0,
            last_tombstone: None,
        }
    }

    #[inline]
    pub fn with_hasher(hash_builder: S) -> Self {
        Self::with_capacity_and_hasher(0, hash_builder)
    }

    pub fn reserve(&mut self, capacity: usize) {
        self.map.reserve(capacity);
        self.values.reserve(capacity);
    }

    #[inline]
    pub fn get_index<Q>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.get(key).map(|i| *i)
    }

    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.contains_key(key)
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match self.get_index(key) {
            Some(index) => {
                self.values.get(index).unwrap().try_value()
            }
            None => None
        }
    }

    pub fn get_mut<Q>(&self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match self.get_index(key) {
            Some(index) => {
                self.values.get_mut(index).unwrap().try_value_mut()
            }
            None => None
        }
    }

    fn replace_bucket(&mut self, index: usize, bucket: Bucket<V>) -> Bucket<V> {
        mem::replace(self.values.get_mut(index).unwrap(), bucket)
    }

    fn replace_value(&mut self, index: usize, bucket: Bucket<V>) -> V {
        self.replace_bucket(index, bucket).into_value()
    }

    fn replace_tombstone(&mut self, index: usize, bucket: Bucket<V>) -> Option<usize> {
        self.replace_bucket(index, bucket).into_tombstone()
    }

    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.insert_index(key, value).1
    }

    pub fn insert_index(&mut self, key: K, value: V) -> (usize, Option<V>) {
        match self.map.entry(key) {
            Entry::Occupied(entry) => {
                // Entry already exists. Just overwrite the value in the vector.
                // Since this entry already exists, we can be sure that the
                // value we're replacing is not a tombstone.
                let index = *entry.get();
                (index, Some(self.replace_value(index, Bucket::Value(value))))
            }
            Entry::Vacant(entry) => {
                match self.last_tombstone {
                    Some(index) => {
                        // Reuse a tombstone.
                        self.last_tombstone = self.replace_tombstone(index, Bucket::Value(value));
                        self.tombstones -= 1;
                        (index, None)
                    }
                    None => {
                        // Insert a new value.
                        let index = self.values.len();
                        self.values.push(Bucket::Value(value));
                        entry.insert(index);
                        (index, None)
                    }
                }
            }
        }
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
        self.map.remove_entry(k).map(|(key, index)| {
            let tombstone = mem::replace(&mut self.last_tombstone, Some(index));
            self.tombstones += 1;
            let value = self.replace_value(index, Bucket::Tombstone(tombstone));
            (key, value)
        })
    }

    pub fn indices(&self) -> Indices<V> {
        Indices {
            iter: self.values.iter().enumerate(),
            tombstones: self.tombstones,
        }
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
    S: BuildHasher
{
    type Output = V;

    #[inline]
    fn index(&self, key: &Q) -> &Self::Output {
        self.get(key).expect("no entry found for key")
    }
}

//impl<'a, K, V, S> IntoIterator for &'a HolyHashMap<K, V, S>
//where
//    K: Eq + Hash,
//    S: BuildHasher,
//{
//    type Item = (&'a K, &'a V);
//    type IntoIter = Iter<'a, K, V>;
//
//    fn into_iter(self) -> Self::IntoIter {
//        self.iter()
//    }
//}

#[derive(Debug, Clone, PartialEq)]
enum Bucket<V> {
    /// A tombstone indicates that the entry has been deleted and can be
    /// reused. The tombstone points to the previous tombstone.
    Tombstone(Option<usize>),

    /// A valid entry.
    Value(V),
}

impl<V> Bucket<V> {
    pub fn value(&self) -> &V {
        match self {
            Bucket::Value(v) => v,
            _ => unreachable!(),
        }
    }

    pub fn try_value(&self) -> Option<&V> {
        match self {
            Bucket::Value(v) => Some(v),
            _ => None,
        }
    }

    pub fn try_value_mut(&mut self) -> Option<&mut V> {
        match self {
            Bucket::Value(v) => Some(v),
            _ => None,
        }
    }

    pub fn into_value(self) -> V {
        match self {
            Bucket::Value(v) => v,
            _ => unreachable!(),
        }
    }

    pub fn into_tombstone(self) -> Option<usize> {
        match self {
            Bucket::Tombstone(v) => v,
            _ => unreachable!(),
        }
    }
}

pub struct Indices<'a, V: 'a> {
    iter: Enumerate<slice::Iter<'a, Bucket<V>>>,
    tombstones: usize,
}

impl<'a, V> Iterator for Indices<'a, V> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((index, entry)) = self.iter.next() {
            match entry {
                Bucket::Value(_) => return Some(index),
                Bucket::Tombstone(_) => {
                    self.tombstones -= 1;
                    continue;
                }
            }
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.tombstones == 0 {
            self.iter.nth(n).map(|(i, _)| i)
        } else {
            let tombstones = &mut self.tombstones;
            self.iter
                .by_ref()
                .filter_map(move |(index, bucket)| match bucket {
                    Bucket::Value(v) => Some(index),
                    Bucket::Tombstone(_) => {
                        *tombstones -= 1;
                        None
                    }
                }).nth(n)
        }
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, V> ExactSizeIterator for Indices<'a, V> {
    fn len(&self) -> usize {
        self.iter.len() - self.tombstones
    }
}

impl<'a, V> DoubleEndedIterator for Indices<'a, V> {
    fn next_back(&mut self) -> Option<usize> {
        while let Some((index, entry)) = self.iter.next_back() {
            match entry {
                Bucket::Value(_) => return Some(index),
                Bucket::Tombstone(_) => {
                    self.tombstones -= 1;
                    continue;
                }
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
            assert!(m.insert(i, i*2).is_none());
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
