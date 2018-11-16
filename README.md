# holyhashmap

[![Build Status](https://api.cirrus-ci.com/github/jasonwhite/holyhashmap.svg?branch=master)](https://cirrus-ci.com/github/jasonwhite/holyhashmap) [![Crates.io](https://img.shields.io/crates/v/holyhashmap.svg)](https://crates.io/crates/holyhashmap)

A hash map whose entries can be indexed into. This is just like [indexmap][],
but the indices are stable (i.e., indices are not perturbed upon removal of an
entry). This makes it an ideal data structure for implementing graphs.

The underlying hash map implementation ([linear probing][]) is not particularly
fast or smart. I'm more interested in the stable indices property than having a
super-fast hash map at the moment. However, I'd be great to switch to either
[Robin Hood Hashing][] or to the [SwissTable approach][hashbrown]. The idea of
stable indices can be applied to any hash map implementation.

[indexmap]: https://github.com/bluss/indexmap
[linear probing]: https://en.wikipedia.org/wiki/Linear_probing
[Robin Hood Hashing]: https://en.wikipedia.org/wiki/Hash_table#Robin_Hood_hashing
[hashbrown]: https://github.com/Amanieu/hashbrown


## Features

 * A drop-in replacement of Rust's
   [`HashMap`](https://doc.rust-lang.org/std/collections/struct.HashMap.html).
 * Inserting a key-value pair gives back an index to refer back to it. Using the
   index bypasses the need to compute the hash of the key.
 * Removing a key-value pair frees up the index that was using it. A future
   insertion will reuse the index. Thus, indices are not compact after a
   removal.


## Usage

Add this to your `Cargo.toml`

```toml
[dependencies]
holyhashmap = "0.1"
```

and this to your crate root:

```rust
extern crate holyhashmap;
```

For [serde](https://serde.rs/) support, add this instead to your `Cargo.toml`:

```toml
[dependencies]
holyhashmap = { version = "0.1", features = ["serde"] }
```

## Example

Here's how one might implement a graph data structure utilizing indices:

```rust
extern crate holyhashmap;

use holyhashmap::{HolyHashMap, EntryIndex};

type NodeIndex = EntryIndex;

struct Neighbors {
    incoming: Vec<NodeIndex>,
    outgoing: Vec<NodeIndex>,
}

pub struct Graph<N, E>
where
    N: Eq + Hash,
{
    // The nodes in the graph. A mapping of the node key `N` to its neighbors.
    nodes: HolyHashMap<N, Neighbors>,

    // The edges in the graph. A mapping between node index pairs and the edge
    // weight `E`.
    edges: HolyHashMap<(NodeIndex, NodeIndex), E>,
}
```

## License

[MIT license](LICENSE)
