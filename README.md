# NLP-Land-prediction-endpoint

<p align="center">
<a href="https://codecov.io/gh/ag-gipp/NLP-Land-prediction-endpoint"><img src="https://codecov.io/gh/ag-gipp/NLP-Land-prediction-endpoint/branch/main/graph/badge.svg?token=7CL6B5LNKP"/></a>    
<a href="https://github.com/ag-gipp/NLP-Land-prediction-endpoint/actions/workflows/release.yml"><img alt="Actions Status" src="https://github.com/ag-gipp/NLP-Land-prediction-endpoint/actions/workflows/release.yml/badge.svg?branch=dev">    
<a href="https://github.com/ag-gipp/NLP-Land-prediction-endpoint/actions/workflows/main.yml"><img alt="Actions Status" src="https://github.com/ag-gipp/NLP-Land-prediction-endpoint/actions/workflows/main.yml/badge.svg?branch=dev">
<a href="https://github.com/ag-gipp/NLP-Land-prediction-endpoint/releases"><img alt="Actions Status" src="https://img.shields.io/github/v/release/ag-gipp/NLP-Land-prediction-endpoint"></a>
<a href="https://ag-gipp.github.io/NLP-Land-prediction-endpoint/"><img alt="Docs" src="https://img.shields.io/badge/Docs-gh--pages-blue"></a>
<a href="https://github.com/ag-gipp/NLP-Land-prediction-endpoint/blob/master/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

## Installation & Setup

### Production

To spin up the production version of this project, switch into the root directory of this project and run:

```console
docker-compose up --build
```

### Development

For development install [poetry](https://python-poetry.org/):

```console
pip install poetry
```

Mac version:

```console
brew install poetry
```

Then run:

```console
poetry install
```

Then you can use the api like this:

```console
poetry run python dev.py
```

If you are using VSCode, you can also run debugging using the `.vscode/launch.json`.

## Code quality and tests

To maintain a consistent and well-tested repository, we use unit tests, linting, and typing checkers with GitHub actions. We use pytest for testing, pylint for linting, and pyright for typing.
Every time code gets pushed to our repository these checks are executed and have to fullfill certain requirements before you can merge the code to our master branch.

We also use naming conventions for branches, commits, and pull requests to leverage GitHub workflow automation and keep the repository clean.

In the following we will describe how to run checks locally and which naming conventions we use.

### CI

Whenever you create a pull request against the default branch, GitHub actions will create a CI job executing unit tests and linting.

### Pre-commit

To make sure the code requirements are satisfied before you push code to the repository, we use pre-commit hooks.

Install the pre-commit hooks using:

```console
poetry run pre-commit install
```

These hooks are automatically checked before you make a commit. To manually run the pre-commit checks, run:

```console
poetry run pre-commit run --all-files
```

### Replicate CI locally

If you want to replicate the exact same pipeline that runs on GitHub actions, install act from [here](https://github.com/nektos/act).

To run the full check suite, execute:

```sh
act -j Test
```

To run a single check from the pipeline such as linting, execute:

```sh
act -j Lint
```

You can also run each of the commands checked in `.github/workflows/main.yml`:

```console
poetry run poe lint
poetry run poe type
poetry run poe test
```

### Repository and naming conventions

Each feature request, bug, enhancement, etc. has to be related to an issue. We have templates for bugs and features requests when you create an issue on GitHub.
An issue should be a closed component that can be implemented by one developer in 1 day. If the issue is larger than that, split it into smaller components.

We group issues using a task list in another issue that has the `Epic` label. These issues are larger components that need to be developed.
Each issue with the `Epic` label has a task list with each element of the task list being a issue (e.g., this one [#47](https://github.com/ag-gipp/NLPLand/issues/47)).
Whenever a pull request with the above convention gets merged, the corresponding issue gets closed, and the task in the Epic gets checked.

When a branch is assigned to you, a new issue will be created from the `dev` branch including the issue number.

To indicate whether the PR is a patch, minor, or major update, please use #patch, #minor, #major in the last commit message of the PR and in the PR description.
See [here](https://github.com/anothrNick/github-tag-action) for more information.

To build changelogs, each pull-request needs one of the labels "fix", "feature", or "test". See [here](https://github.com/mikepenz/release-changelog-builder-action) for more information.

## Contributing

Fork the repo, make changes and send a PR. We'll review it together!

## License

This project is licensed under the terms of MIT license. For more information, please see the [LICENSE](LICENSE) file.

## Citation

If you use this repository, or use our tool for analysis, please cite our work:

TODO: Add citation here and as CITATION.cff file when paper is out.
