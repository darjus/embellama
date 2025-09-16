# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-09-16

### Bug Fixes

- Remove GPU features from docs.rs build configuration ([dad3e9b](https://github.com/darjus/embellama/commit/dad3e9bad772c7b94a378f67331fcf8ad4c413f1))
- Adjust semantic similarity thresholds based on proportional text change ([46f6f69](https://github.com/darjus/embellama/commit/46f6f69bf47f12c20c9960f8ed3b92ee61936f85))

### Features

- Add embedded server support and library interface ([7a4e375](https://github.com/darjus/embellama/commit/7a4e3753f6fef177d8b96caa7651545629d3dc56))

## [0.2.0] - 2025-09-12

### Bug Fixes

- Complete Phase 5 integration test fixes ([f96b479](https://github.com/darjus/embellama/commit/f96b47954ddcee8ed19a9f2fb792af81273e54b8))
- Normalize embeddings and fix error response format for OpenAI compatibility ([20a3299](https://github.com/darjus/embellama/commit/20a329941c7908d96230499036cecc898354a85a))
- Correct test assertion for normalize_embeddings default value ([2b6fdb8](https://github.com/darjus/embellama/commit/2b6fdb82e4c21c414b2d4a96a46c04f85141d368))
- Escape [CLS] in rustdoc comment to fix broken intra-doc link ([7600ef6](https://github.com/darjus/embellama/commit/7600ef664d5e83abfae5a6f7f91b90d0b06aaddf))
- Resolve clippy pedantic warnings and improve code quality ([5749a0e](https://github.com/darjus/embellama/commit/5749a0e3ff1391e867c70309c8513bd631a013ad))
- Resolve property test failures by adding n_ubatch configuration ([b1a21bf](https://github.com/darjus/embellama/commit/b1a21bf7b32ed088f3b4555ebf1321bc0c64603d))
- Add missing CI dependencies for Linux and macOS ([b63dd8b](https://github.com/darjus/embellama/commit/b63dd8be3df09424050e0ee715871511da34d7de))
- Add missing CI dependencies for Linux and macOS ([91cf85f](https://github.com/darjus/embellama/commit/91cf85f1d3770761fc790ccddf3e1a2f4e4adfd8))

### Features

- Implement Phase 1 - server foundation ([59081a7](https://github.com/darjus/embellama/commit/59081a7dead5e1016215dc703b6a07524c93db2e))
- Implement Phase 2 - worker pool architecture ([f7c3a15](https://github.com/darjus/embellama/commit/f7c3a15bc64f94d7e16c439daa7e0fe3c088e587))
- Implement Phase 3 - OpenAI-compatible API endpoints ([a56e8e6](https://github.com/darjus/embellama/commit/a56e8e65ce4571d8c79137629c4f650bfc042f73))
- Implement Phase 4 - Request/Response Pipeline with security fixes ([8e8c3be](https://github.com/darjus/embellama/commit/8e8c3be6f87802de811c4b5bc2a54c242ee51ec9))
- [**breaking**] Add backend feature support for hardware acceleration ([bff6b7c](https://github.com/darjus/embellama/commit/bff6b7c8572205fc1a1fb8141fd9ace0e4ddd6ca))

### Miscellaneous Tasks

- Prepare for v0.1.0 release to crates.io ([9aa780d](https://github.com/darjus/embellama/commit/9aa780d09bac57901293b775a7cc5a70a4811a78))
- Prepare for crates.io publishing ([6b0e2a9](https://github.com/darjus/embellama/commit/6b0e2a9395f87ecec6ac3d622250bddaa41ef0ee))
- Update .gitignore ([51ea9a7](https://github.com/darjus/embellama/commit/51ea9a71bc5eb5478be76f88eacd4ef66e12cf53))
- Cargo fmt ([f51cd8e](https://github.com/darjus/embellama/commit/f51cd8ed3916aead4ddd17e242ebc2d9f3c09c81))
- Cargo clippy -- -W clippy::pedantic --fix ([ca663aa](https://github.com/darjus/embellama/commit/ca663aa8436539ffff2dd0e501e8885512e7d988))
- Update versions ([58668d0](https://github.com/darjus/embellama/commit/58668d0c2be3f648e4cac918fae72543179451c7))
- Clippy fix across features and targets ([6855a5a](https://github.com/darjus/embellama/commit/6855a5a3652a6ea85a4fbe35faf5c731ca93e67c))
- Update deps ([70248d6](https://github.com/darjus/embellama/commit/70248d6fe9aabb41766b8f1f992d5620ccf0e7ba))
- Add pre-commit hooks with uvx/pipx support ([2203b1e](https://github.com/darjus/embellama/commit/2203b1ed32ef344cba6f4473cdd08778c2eb2378))
- Fix fmtcheck ([3abd194](https://github.com/darjus/embellama/commit/3abd1940d1d807b824b426c90751291c758eade7))

### Ci

- Enhance CI/CD pipeline with just commands and model downloads ([52a54e6](https://github.com/darjus/embellama/commit/52a54e65abd01d0a8f3630fa9cde7d4516088ea5))
- Use Ubuntu clang and llvm on linux instead of compiling one ([17f1373](https://github.com/darjus/embellama/commit/17f1373aa9abc799943d67812fc333312daa5d25))

## [0.1.0] - 2025-09-02

### Bug Fixes

- Implement backend singleton to prevent multiple initialization errors ([9ce64ed](https://github.com/darjus/embellama/commit/9ce64ed12ca2291b231221c00cc78d472b9cb96b))

### Documentation

- Separate README.md and DEVELOPMENT.md ([7aa8533](https://github.com/darjus/embellama/commit/7aa853379d5791e6086f0e063218e693aaa164b7))
- Add .clog.toml ([d477333](https://github.com/darjus/embellama/commit/d4773330481af2313f0fbbb06b8632920d2802a5))
- Formatting ([c07dec9](https://github.com/darjus/embellama/commit/c07dec95c253452558ad0838f26335245c57db2a))

### Features

- Implement Phase 1 - Project Setup & Core Infrastructure ([f8ef59c](https://github.com/darjus/embellama/commit/f8ef59c7b9ff60c2557bd3b23e2aae6c3fca675d))
- Implement Phase 2 - Basic Model Management ([6e9f98c](https://github.com/darjus/embellama/commit/6e9f98c83d9275097d28c60ba5764ccf3c54377f))
- Implement Phase 3 - Single Embedding Generation ([47f9839](https://github.com/darjus/embellama/commit/47f98390b1ec505258ea7e987014023fbce17d06))
- Implement Phase 4 - Batch Processing ([3a9988e](https://github.com/darjus/embellama/commit/3a9988e9c889e990870b5b7b98bd33e201444d63))
- Implement Phase 5 - Testing & Documentation ([e92fac2](https://github.com/darjus/embellama/commit/e92fac2c0ac7254248433c0028a50ce96a966373))
- Add comprehensive test infrastructure with real GGUF models ([7dafdbb](https://github.com/darjus/embellama/commit/7dafdbbb76e270bdb22f34302a820f02b9be0477))
- Add n_seq_max configuration and true batch processing ([af0c99f](https://github.com/darjus/embellama/commit/af0c99fa16d530fbd787e697c2460661c1424a9d))

### Miscellaneous Tasks

- Migrate from clog to git-cliff for changelog generation ([1d1066c](https://github.com/darjus/embellama/commit/1d1066ca1d462340818ff8e8436a1d4d586f1170))

### Refactoring

- Move LlamaBackend ownership from model to engine ([08ae8e6](https://github.com/darjus/embellama/commit/08ae8e6fe8ca2c5e983d22e16c81ed961ed316d4))

## [0.0.0] - 2025-08-26

<!-- generated by git-cliff -->
