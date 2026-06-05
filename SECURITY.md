# Security Policy

## Supported versions

This is research-grade software under active development. Only the latest `main`
is supported; there are no backported fixes.

## Reporting a vulnerability

Please report security issues privately via
[GitHub Security Advisories](https://github.com/Lawson-Darrow/Biomedical-RAG-Agent/security/advisories/new)
rather than a public issue. We will acknowledge and respond as soon as we can.

## Scope notes

This agent sends your questions and the retrieved literature passages to whatever
LLM provider you configure (model traffic routes through LLMGateway, an
OpenAI-compatible endpoint). Do not run it on sensitive data with a provider you
do not trust. Provider keys live in `.env`, which is gitignored; never commit real
keys. The default `database_url` points at a local Postgres/pgvector instance, so
change the credentials before exposing it on a network.

This is an evidence-retrieval tool for clinicians and researchers, not a
patient-facing medical device, and must not be used for diagnosis or treatment
decisions.
