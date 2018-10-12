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

extern crate serde;

use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

use self::serde::de::{self, Deserialize, Deserializer, MapAccess, Visitor};
use self::serde::ser::{Serialize, SerializeMap, Serializer};

use {HolyHashMap, EntryIndex};

impl<K, V, S> Serialize for HolyHashMap<K, V, S>
where
    K: Serialize + Hash + Eq,
    V: Serialize,
    S: BuildHasher,
{
    fn serialize<T>(&self, serializer: T) -> Result<T::Ok, T::Error>
    where
        T: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.len()))?;
        for (k, v) in self {
            map.serialize_entry(k, v)?;
        }

        map.end()
    }
}

struct MapVisitor<K, V, S>(PhantomData<(K, V, S)>);

impl<'de, K, V, S> Visitor<'de> for MapVisitor<K, V, S>
where
    K: Deserialize<'de> + Hash + Eq,
    V: Deserialize<'de>,
    S: Default + BuildHasher,
{
    type Value = HolyHashMap<K, V, S>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "a map")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = HolyHashMap::with_capacity_and_hasher(
            map.size_hint().unwrap_or(0),
            S::default(),
        );

        while let Some((key, value)) = map.next_entry()? {
            values.insert(key, value);
        }

        Ok(values)
    }
}

impl<'de, K, V, S> Deserialize<'de> for HolyHashMap<K, V, S>
where
    K: Deserialize<'de> + Hash + Eq,
    V: Deserialize<'de>,
    S: Default + BuildHasher,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(MapVisitor(PhantomData))
    }
}

impl Serialize for EntryIndex
{
    fn serialize<T>(&self, serializer: T) -> Result<T::Ok, T::Error>
    where
        T: Serializer,
    {
        serializer.serialize_u64(self.0 as u64)
    }
}

struct IndexVisitor;

impl<'de> Visitor<'de> for IndexVisitor {
    type Value = EntryIndex;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "an entry index")
    }

    fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(EntryIndex(v as usize))
    }
}

impl<'de> Deserialize<'de> for EntryIndex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_u64(IndexVisitor)
    }
}

#[cfg(test)]
mod test {
    extern crate serde_test;

    use HolyHashMap;

    use self::serde_test::{assert_tokens, Token};

    #[test]
    fn test_empty() {
        let map = HolyHashMap::<u32, u32>::new();

        assert_tokens(&map, &[Token::Map { len: Some(0) }, Token::MapEnd]);
    }

    #[test]
    fn test_non_empty() {
        let mut map = HolyHashMap::new();
        map.insert(1, 1);
        map.insert(2, 4);
        map.insert(3, 9);

        assert_tokens(
            &map,
            &[
                Token::Map { len: Some(3) },
                Token::I32(1),
                Token::I32(1),
                Token::I32(2),
                Token::I32(4),
                Token::I32(3),
                Token::I32(9),
                Token::MapEnd,
            ],
        );
    }
}
