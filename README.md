
Overall Planned Structure:

services/api/app/
├── main.py
├── core/
│   ├── database.py
│   └── config.py              # all env vars and settings in one place
├── models/
│   └── models.py              # SQLAlchemy tables
├── schemas/
│   └── schemas.py             # Pydantic request/response shapes
├── routers/
│   ├── health.py
│   ├── models.py              # CRUD for model registry
│   └── inference.py           # predict + predict/async + jobs + stats
└── inference/
    ├── base.py                # the abstract base class (the interface)
    ├── registry.py            # maps backend_type string → backend class
    └── backends/
        ├── pickle_backend.py  # week 1
        ├── onnx_backend.py    # week 2
        └── remote_backend.py  # week 3
