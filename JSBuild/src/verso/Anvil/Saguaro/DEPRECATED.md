# Deprecated Packaging Layout

`Saguaro/` is retained as a legacy archive/reference directory.

Active runtime package location is now:

- `saguaro/` (top-level package in the Anvil repository)

Anvil integrations should import directly from the top-level package, for example:

```python
from saguaro.api import SaguaroAPI
```

Do not depend on `Saguaro/setup.py` or `pip install -e Saguaro` for normal in-repo operation.
