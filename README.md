# Kivi

Kivi is an experimental distributed state system written in Rust

The project explores what a modern Redis-like system could look like if it was
designed around current hardware, distributed systems, and explicit
semantics instead of accumulating another decade of compatibility constraints

Kivi is built around a fairly simple idea: logical state and physical
representation should not be the same thing

That means the system is free to adapt how data is partitioned, replicated,
materialized, stored, moved, cached, or accelerated without changing what that
state means to the application

The architecture focuses on:

- low and predictable latency
- adaptive partitioning and placement
- explicit consistency and durability semantics
- horizontal scaling without fixed Redis-style slot architecture
- deterministic and testable correctness paths
- the ability to take advantage of newer storage, memory, and networking
  technologies without redesigning the logical model every time hardware
  becomes interesting again

This is not "Redis but rewritten in Rust" There are already enough projects
whose architectural roadmap is essentially that sentence

Kivi is still a research-oriented project and is under active development

The full architecture lives in [`docs/rfc.md`](docs/rfc.md).

## Status

Early development

Kivi is not ready for production use, and relying on it for anything important
would currently be an unusually creative operational decision

## Contributing

Contributions, experiments, reviews, benchmarks, criticism, and arguments about
distributed systems are welcome.

You do not need to commit to becoming a maintainer. If you find one part of the
system interesting and want to improve it, test it, break it, or explain why it
is wrong, that is useful too

## License

Core database components are licensed under AGPL-3.0-or-later

Client, protocol, interoperability, and selected tooling crates are licensed
under Apache-2.0
