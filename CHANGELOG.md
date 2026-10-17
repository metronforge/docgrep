# Changelog

<!-- version list -->

## v1.21.1 (2026-10-17)

### Bug Fixes

- **usecases**: Skip empty sparse vectors when re-encoding
  ([`4f87673`](https://github.com/metronforge/docgrep/commit/4f87673796c4cff316e2f84469b7740800d4aa6c))


## v1.21.0 (2026-10-17)

### Documentation

- Describe BM25 sparse weights and the rerank pool
  ([`c8bc97d`](https://github.com/metronforge/docgrep/commit/c8bc97de658114df1604a6edc5fe9d2e485dbc24))

### Features

- **adapters**: Rerank a configurable pool of fused candidates
  ([`6eadd2b`](https://github.com/metronforge/docgrep/commit/6eadd2b67aa6713914b00be72dafef24fd4792b9))

- **adapters**: Weigh sparse vectors with BM25
  ([`16c31c9`](https://github.com/metronforge/docgrep/commit/16c31c9082f45f50f3765d38e3d763945c3e795e))

- **api**: Add reencode-sparse and the rerank pool to eval and status
  ([`dc40c60`](https://github.com/metronforge/docgrep/commit/dc40c603ce3fa30735e88b938857f4f7e049c8b8))

- **usecases**: Re-encode stored sparse vectors
  ([`10a067c`](https://github.com/metronforge/docgrep/commit/10a067ce03f040312ca58178e6e2df2bf4b32d55))


## v1.20.0 (2026-10-17)

### Documentation

- Describe the retrieval evaluation harness
  ([`0ed12dd`](https://github.com/metronforge/docgrep/commit/0ed12dd55bd3a26d6131bcfd6c8dc4de75f7555a))

### Features

- **adapters**: Probe the dense, sparse and fused channels
  ([`01cb309`](https://github.com/metronforge/docgrep/commit/01cb3091ef55acd75aaa68b0997012fa6d4f88fa))

- **api**: Add the docgrep eval command
  ([`50638b3`](https://github.com/metronforge/docgrep/commit/50638b32963ba57c473d3136ceebbc0f61719ff1))

- **usecases**: Evaluate known-item retrieval cases
  ([`146cae8`](https://github.com/metronforge/docgrep/commit/146cae85710e8a765ecaa024fc3e546f6c39fae1))

- **usecases**: Expose the full ranked pipeline for diagnostics
  ([`0e6c033`](https://github.com/metronforge/docgrep/commit/0e6c033ac54574f0ebb7ddc7b6c7eb22a453fd51))


## v1.19.0 (2026-10-17)

### Documentation

- Describe corpus roots
  ([`b9a4431`](https://github.com/metronforge/docgrep/commit/b9a4431c32ba82505fd09f14c140a40a048eefc8))

### Features

- **adapters**: Resolve stored paths through the corpus roots
  ([`46fc750`](https://github.com/metronforge/docgrep/commit/46fc750ca06e5ade13075229e1850d63f8fb7796))

- **api**: Roots commands, root parameters, root filter and per-root status
  ([`b8bf2e0`](https://github.com/metronforge/docgrep/commit/b8bf2e0219c14aca0e67772aafed5ba56f99c752))

- **domain**: Model corpus roots and their stored identity
  ([`3125b88`](https://github.com/metronforge/docgrep/commit/3125b889d380e981f0e4b887e7ccba20dc6153dd))

- **settings**: Store several corpus roots in a roots file
  ([`2253527`](https://github.com/metronforge/docgrep/commit/225352745fbe129f9a7c3d9fb2240b0dcc33ce28))

- **usecases**: Index, update and report across roots
  ([`6858b6e`](https://github.com/metronforge/docgrep/commit/6858b6e3a8537f1cb4d493a16526cef33e3c19a8))


## v1.18.0 (2026-10-17)

### Documentation

- Describe passage context
  ([`6108296`](https://github.com/metronforge/docgrep/commit/6108296d168a610c07c2f4ccfbbddebd6aad4ff4))

### Features

- **adapters**: Read passages and source outlines for context
  ([`0ccfeb8`](https://github.com/metronforge/docgrep/commit/0ccfeb82ee12576762e84e9dd5bb5ddc9accf0b5))

- **api**: Expand passage context over HTTP, MCP and the CLI
  ([`2cb172f`](https://github.com/metronforge/docgrep/commit/2cb172fbe6e8cdbc13473b9df9a6c966e56c36b5))

- **domain**: Assemble passage context from neighbouring passages
  ([`a1ca54d`](https://github.com/metronforge/docgrep/commit/a1ca54d54df5212be817755e5fec8bdd3855f7c5))

- **usecases**: Expand the context around passages
  ([`4b0ba77`](https://github.com/metronforge/docgrep/commit/4b0ba770b4e4a6f00b2122e9ba74c2dc0d9e88ac))


## v1.17.0 (2026-10-17)

### Documentation

- Describe metadata sanitizing
  ([`de557b7`](https://github.com/metronforge/docgrep/commit/de557b7c5ddca8ddc09e2afcb93cbcef94390cb1))

### Features

- **api**: Expose dropped-metadata audit and filter
  ([`ad8e93a`](https://github.com/metronforge/docgrep/commit/ad8e93a0d4edacbdfca85b8027175bbad8ab69a2))

- **domain**: Sanitize implausible source metadata
  ([`4fa8894`](https://github.com/metronforge/docgrep/commit/4fa88942d174cbb6c4b9edae3d1a614ff8c13995))

- **usecases**: Apply sanitized metadata to filters, browse and search results
  ([`066b326`](https://github.com/metronforge/docgrep/commit/066b326bd6e4c7288aa5b07514d5ca56b5c6938d))


## v1.16.0 (2026-10-17)

### Documentation

- Describe filtered search
  ([`c0d61b6`](https://github.com/metronforge/docgrep/commit/c0d61b6871785db312d6c9d688cd2d57a5f6feb4))

### Features

- **adapters**: Restrict first-stage retrieval to an allow-list of sources
  ([`2fa8409`](https://github.com/metronforge/docgrep/commit/2fa8409c1165a275eb7dd80650d99441b621a417))

- **api**: Structured filters in /search, find_passages and the CLI
  ([`84dfc40`](https://github.com/metronforge/docgrep/commit/84dfc401044d2b193afd04a5afc182e7c8581b4a))

- **usecases**: Filter search by source metadata
  ([`706912b`](https://github.com/metronforge/docgrep/commit/706912b7cbd6421be410decd8762d21c0fdc3e5a))


## v1.15.0 (2026-10-16)

### Documentation

- Describe source browsing and the shared filter
  ([`5123b3c`](https://github.com/metronforge/docgrep/commit/5123b3c486c0397cd6a3f40bd9a4bdf69fd106c7))

### Features

- **api**: Browse sources over HTTP, MCP and the CLI
  ([`499900d`](https://github.com/metronforge/docgrep/commit/499900da5094f7b76b2379e14d374479330c22d2))

- **domain**: Add the shared source filter and browse models
  ([`25839cd`](https://github.com/metronforge/docgrep/commit/25839cd04ed9a7d0d9732a707e380e7269cd2b80))

- **usecases**: Browse indexed sources
  ([`9b87ccd`](https://github.com/metronforge/docgrep/commit/9b87ccdb168ebc260e7d36bc613acf03febadae0))


## v1.14.1 (2026-10-16)

### Bug Fixes

- **api**: Declare output schemas for the MCP tools
  ([`74f2b8e`](https://github.com/metronforge/docgrep/commit/74f2b8e2f441c859dc289c2eeb76452490fcdbd7))

- **api**: Report invalid settings with the documented exit status
  ([`3e145f9`](https://github.com/metronforge/docgrep/commit/3e145f99c11e9bbebb3e110c6b08d94cc0c2799a))

- **api**: Report the docgrep version as the MCP server version
  ([`0e023a8`](https://github.com/metronforge/docgrep/commit/0e023a82765c209b394daa3ea681eed6d25c4a96))

- **api**: Return malformed MCP tool arguments as validation error bodies
  ([`f8071c5`](https://github.com/metronforge/docgrep/commit/f8071c5993f6bb0203d5fdd1255ffa78ede42c37))

### Documentation

- Note that local-mode status can report the store as busy after a search
  ([`8dca1b6`](https://github.com/metronforge/docgrep/commit/8dca1b616c25d670087b2ca4bdf04db93b88ad76))


## v1.14.0 (2026-10-15)

### Build System

- **deps**: Add the MCP Python SDK
  ([`9d3b1e2`](https://github.com/metronforge/docgrep/commit/9d3b1e2f22b351a4336b9a19d5d3d909ac609b08))

### Documentation

- Record the MCP front-end decision and document the MCP server
  ([`27adf71`](https://github.com/metronforge/docgrep/commit/27adf71062f3b1e8f1e67f409e62f95d8fbf8b1a))

### Features

- **api**: Serve read-only MCP tools over stdio and streamable HTTP
  ([`44b7d00`](https://github.com/metronforge/docgrep/commit/44b7d0014a58c10c98985768a3e1cdd2c9d46848))


## v1.13.1 (2026-10-15)

### Bug Fixes

- **usecases**: Persist parse failures in the manifest
  ([`aec1681`](https://github.com/metronforge/docgrep/commit/aec16818985fccb7baffa3be001c9ea5c866c0f5))

### Documentation

- Note how an update records failed sources in ADR-0013
  ([`7a8efa0`](https://github.com/metronforge/docgrep/commit/7a8efa07a19cf3a0f5ad03b902679e1b618a48af))


## v1.13.0 (2026-10-15)

### Documentation

- Add ADR for DOCX parsing
  ([`60d199b`](https://github.com/metronforge/docgrep/commit/60d199b002f8101a1fafaf1ed0a6ac2123b1eb2d))

### Features

- **adapters**: Add DOCX parser
  ([`cf5c4a2`](https://github.com/metronforge/docgrep/commit/cf5c4a2ff1ec2e2e02cfcd60a065a49aee3d890f))


## v1.12.0 (2026-10-07)

### Documentation

- Record the OCR decision and document the ocr command
  ([`42b8f4a`](https://github.com/metronforge/docgrep/commit/42b8f4a1c8850e702012fde22a4060b6ee6e2799))

### Features

- **adapters**: Add the OCR registry and sibling delegation tables
  ([`be7ad2d`](https://github.com/metronforge/docgrep/commit/be7ad2d74652cc5a0138fc42120e1f8a2804af42))

- **usecases**: Index OCR outputs instead of their delegated originals
  ([`ebf9c99`](https://github.com/metronforge/docgrep/commit/ebf9c996d2544c42a973f2e48aa0aa5f00bb1858))

- **usecases**: Process selected scans through OCR with ocrmypdf
  ([`f2995a8`](https://github.com/metronforge/docgrep/commit/f2995a8480138c71b959efd7fabd89de50d7f459))


## v1.11.1 (2026-10-07)

### Bug Fixes

- **adapters**: Keep absolute paths and raw tool output out of failure messages
  ([`89dca7c`](https://github.com/metronforge/docgrep/commit/89dca7c77015b91fa9957224e9690c680ae3067b))

- **cli**: Ignore a standard output pipe closed by its reader
  ([`ef96e55`](https://github.com/metronforge/docgrep/commit/ef96e55da619240fd9b42514a83451147a182504))

### Documentation

- Point the server-mode test example at a throwaway Qdrant container
  ([`449f41e`](https://github.com/metronforge/docgrep/commit/449f41ebcf320c00223967313d6cb03d809d7679))

### Testing

- Refuse server-mode tests on a Qdrant server holding other data
  ([`b8503cb`](https://github.com/metronforge/docgrep/commit/b8503cbdd08228b407e028f0c7be32b0fea90eba))


## v1.11.0 (2026-10-05)

### Documentation

- Add ADR-0016 on collapsing exact duplicate passages at query time
  ([`cce8484`](https://github.com/metronforge/docgrep/commit/cce8484c68af7893048d134f8d19ddf9ce9d63f1))

- Add how-to examples to the README
  ([`2750726`](https://github.com/metronforge/docgrep/commit/2750726bce7b5905f3d5709be860adca862ad781))

### Features

- **usecases**: Collapse passages with identical text from several files
  ([`96dcc5f`](https://github.com/metronforge/docgrep/commit/96dcc5f965a9ffac6915650a265a4a08d4cab858))


## v1.10.0 (2026-10-05)

### Documentation

- Add ADR-0015 on reading DjVu through djvulibre subprocesses
  ([`f03dc08`](https://github.com/metronforge/docgrep/commit/f03dc08b575938fcec401cb38d4d4fa61a88dcf6))

### Features

- **adapters**: Read DjVu text layers with the djvulibre tools
  ([`9e4a2df`](https://github.com/metronforge/docgrep/commit/9e4a2dfda86527f98469da10e9bc234c250d69ac))

- **api**: Discover PDF and DjVu files that need OCR
  ([`74caff5`](https://github.com/metronforge/docgrep/commit/74caff523066e60422ad63da79fac7f64f911e19))


## v1.9.0 (2026-10-04)

### Documentation

- Note how to refresh the editable install after a release
  ([`9495d73`](https://github.com/metronforge/docgrep/commit/9495d7323047f69e075e83cd437143e399a8f4a2))

### Features

- **cli**: Show a breakdown of the sources table in docgrep status
  ([`16db8c3`](https://github.com/metronforge/docgrep/commit/16db8c319f008fad86619fe840e3eb53d07bbe3b))

### Testing

- **adapters**: Spell invisible and look-alike characters as escapes
  ([`f75d2dc`](https://github.com/metronforge/docgrep/commit/f75d2dc767287a6531b53e79cee8e1b6bdffd30a))


## v1.8.0 (2026-10-04)

### Bug Fixes

- **adapters**: Drop PDF pages only for broken characters
  ([`106ea63`](https://github.com/metronforge/docgrep/commit/106ea639ca54267612a2f40ff03c8b6161d9b2d6))

### Features

- **cli**: Show which page rules fire in the dry-run sample
  ([`c3ed844`](https://github.com/metronforge/docgrep/commit/c3ed8449c3da837189923f9b92cf483edef44dde))


## v1.7.0 (2026-10-04)

### Chores

- Relicense under AGPL-3.0-or-later
  ([`b4a11f3`](https://github.com/metronforge/docgrep/commit/b4a11f31807220e96afb2114479ba0d76efa973b))

### Documentation

- Add ADR-0014 on the AGPL-3.0-or-later license
  ([`d9c83e3`](https://github.com/metronforge/docgrep/commit/d9c83e36b535fdea5e6ade1e04ddd3f78c677f90))

### Features

- **adapters**: Remove soft hyphens and drop garbled PDF text layers
  ([`711419b`](https://github.com/metronforge/docgrep/commit/711419bb3abb41ab18f9916ebbd727f8d730ce2e))

- **cli**: Measure PDF text layers in the dry-run sample
  ([`019ba7e`](https://github.com/metronforge/docgrep/commit/019ba7e4ab6627c3d1baf269d5bef7dc58cc91db))

- **usecases**: Give parsers versions that drive re-extraction on update
  ([`5d5769b`](https://github.com/metronforge/docgrep/commit/5d5769b3c7542ab936d4da194492dc6a83f4b9df))


## v1.6.0 (2026-10-04)

### Documentation

- Add ADR-0013 on incremental update writing in place
  ([`0000fda`](https://github.com/metronforge/docgrep/commit/0000fda4f481481a58fcf7258303f8cfade4faa3))

### Features

- **adapters**: Parse EPUB books with the standard library
  ([`7966066`](https://github.com/metronforge/docgrep/commit/7966066f16ed2540a38673c6686a231668ed9f69))

- **cli**: Add docgrep update and show updates in status
  ([`992f88c`](https://github.com/metronforge/docgrep/commit/992f88c9d23c014afee317730abca0ea6bc657c1))

- **usecases**: Update the published index in place for corpus changes
  ([`c903538`](https://github.com/metronforge/docgrep/commit/c9035387cf3efd30ec5290dc28542bc4c27616f4))

### Testing

- **settings**: Isolate tests from the caller's DOCGREP_ environment
  ([`1a9b32e`](https://github.com/metronforge/docgrep/commit/1a9b32efb55fce845b60f9a5fb4cf3809e5f9250))


## v1.5.1 (2026-10-03)

### Bug Fixes

- **api**: Make POST /index/build refuse to discard a resumable run
  ([`774f0c9`](https://github.com/metronforge/docgrep/commit/774f0c95452f02eabe5ac3ac7cc94b7202efc917))

- **usecases**: Finish a run that died between the alias swap and its status
  ([`117de74`](https://github.com/metronforge/docgrep/commit/117de74d3a556410fb006808ac02133ada892515))


## v1.5.0 (2026-10-03)

### Features

- **cli**: Add a read-only status command
  ([`427f8b9`](https://github.com/metronforge/docgrep/commit/427f8b906f626cf22acbb4e44ba18e315821388b))

- **cli**: Cancel a running index build cooperatively
  ([`d64359f`](https://github.com/metronforge/docgrep/commit/d64359f74d3043d72e9f09db3b18a74447a0df55))

- **cli**: Resume an interrupted index build
  ([`f201c86`](https://github.com/metronforge/docgrep/commit/f201c86b5a50c7a3b5e94d8a396e4b0d7c0da8df))

- **usecases**: Add a persisted run state machine and keep staging on abort
  ([`6fe0095`](https://github.com/metronforge/docgrep/commit/6fe009554a24871b3a291911848630aec50aef5b))

- **usecases**: Log build progress with counters and an ETA
  ([`d2867e5`](https://github.com/metronforge/docgrep/commit/d2867e5eb019697a9a5df061f938e315b979214d))


## v1.4.0 (2026-10-03)

### Features

- **adapters**: Run the embedding model in half precision on CUDA
  ([`97d0e71`](https://github.com/metronforge/docgrep/commit/97d0e7176e86269e98b320f930bb262067f2c7e3))

- **adapters**: Write passages in batches
  ([`e2e872b`](https://github.com/metronforge/docgrep/commit/e2e872b1880e4df273abea4b12862792be000f10))

- **cli**: Add a read-only dry-run command for the corpus (UC-19)
  ([`75cc12e`](https://github.com/metronforge/docgrep/commit/75cc12e5d397fdbfeae1c41571c204f17285edcc))


## v1.3.0 (2026-10-03)

### Documentation

- Add ADR-0012 on durable run state
  ([`2cfc2be`](https://github.com/metronforge/docgrep/commit/2cfc2bef9133491316aaffa89e0d21f1bd0cb6fd))

### Features

- **adapters**: Keep durable run state and clean up staging collections
  ([`7e7649e`](https://github.com/metronforge/docgrep/commit/7e7649e7f343cfc97882cbe597529a56da2475e2))

- **adapters**: Run the reranker in half precision on CUDA
  ([`4c6d87d`](https://github.com/metronforge/docgrep/commit/4c6d87d158aceeccf0f02c34805171411cb4ef00))


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
