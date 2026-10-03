# Changelog

<!-- version list -->

## v1.2.0 (2026-10-03)

### Build System

- **deps**: Select the torch backend through cpu and cu130 extras
  ([`1d29f84`](https://github.com/metronforge/docgrep/commit/1d29f84f00ee1c42d170392a782914075105af13))

### Chores

- **settings**: Move the allowed hosts comment to its field
  ([`98c46d7`](https://github.com/metronforge/docgrep/commit/98c46d78ca2e64a5c93793a107ccf48372a9b0f0))

### Continuous Integration

- Run the Qdrant server tests and document the server setup
  ([`886ba82`](https://github.com/metronforge/docgrep/commit/886ba82b63e15085af7853327295fde32bb7a13f))

### Documentation

- Add ADR-0010 and ADR-0011
  ([`e30de11`](https://github.com/metronforge/docgrep/commit/e30de11840b92ae887da1cecb51d92095187e4bf))

- Amend ADR-0011 to the cpu and cu130 extras
  ([`5a8d3a5`](https://github.com/metronforge/docgrep/commit/5a8d3a513a3f66cd1e41738ee9d460fc02c311aa))

### Features

- **adapters**: Make the Qdrant alias configurable
  ([`16c80f1`](https://github.com/metronforge/docgrep/commit/16c80f12faf61a80f1b9d85ec342d652917403b4))

- **adapters**: Support a Qdrant server next to local mode
  ([`983b59c`](https://github.com/metronforge/docgrep/commit/983b59cf1e41491f5c28ce52acaa2098a6d9738f))


## v1.1.0 (2026-10-03)

### Bug Fixes

- **adapters**: Anchor glob patterns that start with a slash
  ([`a960543`](https://github.com/metronforge/docgrep/commit/a96054352cef5bcdaddd50da64c82f2f8188e875))

- **adapters**: Use the renamed sentence-transformers dimension accessor
  ([`02eacf2`](https://github.com/metronforge/docgrep/commit/02eacf26c98362ecfe9fd3c7d2eb80f7d38aa832))

### Chores

- **deps**: Sync uv.lock with the 1.0.0 release
  ([`7f938ee`](https://github.com/metronforge/docgrep/commit/7f938eeb40b4a00f995800a851d17810f00458bc))

### Continuous Integration

- Refresh uv.lock when a release bumps the version
  ([`f9244a6`](https://github.com/metronforge/docgrep/commit/f9244a640b79d6084e594c7b64ba521599798345))

- Use checkout v5 and the ubuntu-24.04 runner
  ([`e1f0efa`](https://github.com/metronforge/docgrep/commit/e1f0efae43b8d3e5d31c61a62617c30633e5c8c5))

### Documentation

- Explain score when reranking is skipped
  ([`d54e1e9`](https://github.com/metronforge/docgrep/commit/d54e1e933068be490d10b8fa5a24a8343b87352c))

### Features

- **api**: Log index, search, model and server lifecycle at INFO
  ([`688ef5e`](https://github.com/metronforge/docgrep/commit/688ef5e957248c6f3c8b7d5f0f4b2d5490f3bfb0))


## v1.0.0 (2026-10-03)

- Initial Release
