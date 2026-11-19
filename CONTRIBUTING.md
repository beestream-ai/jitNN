# Contributing to jitNN

Thank you for your interest in jitNN. Bug reports, fixes, new operators, documentation and
examples are all welcome. By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before you start

- For anything larger than a small fix, open an issue first so we can agree on the approach.
- [INSTALL.md](docs/INSTALL.md) explains how to build jitNN and set up the test references, and
  [DEVELOPMENT.md](docs/DEVELOPMENT.md) walks through the code, including how to add an operator.
- Run the tests before you open a pull request:

  ```bash
  ./mvnw test                                          # reference backend, no GPU needed
  ./mvnw test -Paccel-tests -Djitnn.test.device=0:0    # on a TornadoVM device, if you have one
  ```

  [VALIDATION.md](docs/VALIDATION.md) describes what the tests check.

## Contributor License Agreement

Before we can merge your first pull request, you need to sign the
[Beestream Contributor License Agreement](CLA.md) (CLA). You keep the copyright in your work. The
CLA grants Beestream Ltd. and the people who receive jitNN a copyright and patent license to your
contributions, on the same terms as the Apache Software Foundation's CLA. The same CLA covers all
of Beestream's open-source projects.

### Signing as an individual

1. Open your pull request. The **CLA assistant** (cla-assistant.io) comments on it if any commit
   author has not signed.
2. Follow the link in that comment, read the CLA, sign in with the GitHub account that made the
   commits, fill in your full name and e-mail address, and accept.
3. The **license/cla** check turns green. You sign once; later pull requests to this repository
   pass automatically. Each Beestream repository keeps its own record and asks you to sign the
   first time you contribute to it, so that you confirm the CLA for each project you join.

If the check does not update, comment `@cla-assistant check` on the pull request.

Every commit author must have signed, and commits must be made with an e-mail address linked to
the author's GitHub account; otherwise the check cannot match the commit to a signature.

### Signing on behalf of a company

If your employer or a client may own the rights in your work, they must sign the
[Corporate CLA](CCLA.md): an authorised signatory fills it in, lists the employees who may
contribute in Schedule A, and emails the signed PDF to info@beestream.ai. Each listed employee
still signs the individual CLA on their pull request as described above.

## Pull requests

- Keep each pull request to one change, with tests for new behaviour.
- New source files carry the Apache 2.0 license header used by the existing files.
- Do not add third-party code without saying where it comes from and under which license; it has
  to be compatible with Apache 2.0 and listed in [NOTICE](NOTICE).
