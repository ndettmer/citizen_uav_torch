# CitizenUAV Python

## Quirks
- for using WeightWatcher
  - Right now the `pyRMT` is broken for the latest python version (see [Github PR](https://github.com/GGiecold/pyRMT/pull/2)). 
  - Workaround: 
    1. clone `pyRMT` and change the import of the collection classes like so: `from collections import MutableSequence, Sequence` -> `from collections.abc import MutableSequence, Sequence`
    2. then install `pyRMT` from your local clone with `pip install -e /path/to/pyRMT`
    3. Don't forget the `--upgrade` flag if you already installed it along with WW