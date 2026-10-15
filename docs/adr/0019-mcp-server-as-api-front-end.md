# ADR-0019: MCP server as a third front-end of the API layer

**Status:** Accepted
**Date:** 2026-10-15

## Context

docgrep returns verbatim passages through two front-ends, the command line and a local HTTP API. Both are thin layers over the same use cases, wired by one composition root. LLM clients such as coding assistants talk to tools through the Model Context Protocol (MCP). Such clients run either a local process that speaks MCP over standard input and output, or connect to an HTTP endpoint.

## Decision

MCP is a third front-end in the API layer, in this repository and package. It is not a separate repository or process: it needs the same composition root and, over HTTP, the same loaded models as the REST API, and a separate package would duplicate the wiring and drift from it. The module `docgrep.api.mcp` builds the server from the operations that the composition root returns, as the HTTP and CLI fronts do. It has no business logic and no Protocols of its own, and nothing in the domain or use-case layers depends on it. Only the API layer imports the SDK, and a test enforces this.

**SDK.** The official MCP Python SDK (package `mcp`) is a regular runtime dependency. It is used through its high-level server API.

**Transports.**
- `docgrep mcp` serves standard input and output for clients that start the server themselves. Standard output belongs to the protocol: the server writes to a duplicate of the original descriptor, descriptor 1 is pointed at standard error, and logs go to standard error only. Models load on the first search, not at startup, so the handshake is fast.
- `docgrep serve` mounts streamable HTTP at `/mcp` of the existing FastAPI app, with its session manager started in the app's lifespan. This is the transport for daily use. One process holds the models for REST and MCP clients alike. Each stdio process would load its own copy, pay the model load on its first search, and in local mode would have to compete for the store with every other process.
- `DOCGREP_MCP_HTTP_ENABLED` turns the mount off.

**Tools.** The tools are read-only and shaped by the domain, one per retrieval use case that exists in the code: `find_passages` (UC-1, UC-2) and `get_index_status` (UC-7). Context expansion (UC-12), citation formatting (UC-13) and source browsing (UC-8) become tools once their use cases exist. Every tool carries the read-only annotation hints. Ingestion, indexing, OCR, configuration and cancel tools are deliberately absent, and no flag enables them: a client model that can change the index or the library needs its own decision.

**Results.**
- Structured results are the bodies that the other fronts already return: the `SearchResponse` model of `POST /search` (reused, not redefined) and the report of `docgrep status --json`. Each comes with a compact plain-text rendering.
- Passage text is never altered. A result over a size budget leaves out the lowest-ranked passages and says so; it never cuts a passage. `top_n` is capped (`DOCGREP_MCP_MAX_TOP_N`, default 25, within `DOCGREP_TOP_N_MAX`) so that default results stay well under the result limits of common clients.
- A passage without a page, such as one from DOCX or EPUB, has a null page and no page in the text.
- Domain errors become tool errors with the category, message and details of an HTTP error body, without tracebacks or absolute paths.

**DNS rebinding.** The HTTP endpoint keeps the bind rule of the HTTP API: 127.0.0.1 by default, no authentication. The app's existing Host and Origin check applies to `/mcp`. The SDK's own DNS-rebinding protection is enabled as well, allowing only 127.0.0.1, localhost, ::1, the configured bind host and `DOCGREP_ALLOWED_HOSTS`. A web page that resolves its own name to 127.0.0.1 cannot drive the tools from a browser.

**Trust.** Passage text comes from the user's own documents and is untrusted data for the client model. A document can contain text that reads like instructions. The server instructions and the tool descriptions tell clients to treat results as data and to cite the source, the heading path and, when present, the page.

**Degradation.** The SDK is imported lazily. Without it, `docgrep mcp` exits with 2 and names the package and the install command, and `docgrep serve` starts without `/mcp` and logs one warning.

## Consequences

LLM clients get the same findings as REST and CLI users, with the same field names, so the fronts cannot drift. The SDK and its dependencies join the lock. Over HTTP the models are shared; over stdio each client process loads its own. In local mode a stdio server and `docgrep serve` cannot both open the store, as with any two docgrep processes, so a Qdrant server is the setup for several clients.

No output schema is declared for the tools, because the SDK validates every returned result against a declared output model, error results included, and that would replace the structured error body. Clients still receive the structured content.

Adding a tool means adding its use case first. A tool that writes would need a new decision on authorization and confirmation.

## Alternatives considered

- **A separate repository or companion process** (like the LLM bridge of ADR-0004) that calls the REST API: it would need its own deployment, would duplicate the request and response models, and would add a network hop. In return it would gain nothing, because the MCP layer holds no LLM inference.
- **Only stdio:** simplest for clients to configure, but every client session would load the models again.
- **Only HTTP:** one process, but clients that can only start a local command could not connect.
- **Generic tools, such as a raw query against the store:** they would expose storage details, bypass the ranking and duplicate rules of the use cases, and make results depend on the client model's queries.
