# docs/

This folder contains all formal documentation produced for the Reading4All capstone project. All documents are written in LaTeX and compiled to PDF. Each subfolder corresponds to a specific deliverable or document type.

---

## Folder Structure

| Folder | Document Type |
|---|---|
| `Checklists/` | Capstone milestone checklists (code, dev plan, SRS, VnV, writing, etc.) |
| `Design/` | Software architecture and detailed design documents |
| `DevelopmentPlan/` | Project development plan |
| `Extras/` | Supplementary deliverables — ML report and usability testing |
| `HazardAnalysis/` | Hazard analysis document |
| `Presentations/` | Presentation slides and demo materials from all project milestones |
| `ProblemStatementAndGoals/` | Problem statement and project goals |
| `ReflectAndTrace/` | Reflection and traceability document |
| `SRS-Volere/` | Software Requirements Specification (SRS) using the Volere template |
| `VnVPlan/` | Verification and Validation Plan |
| `VnVReport/` | Verification and Validation Report (Test Report) |
| `projMngmnt/` | Team contribution reports across project milestones |

Root-level files:

| File | Description |
|---|---|
| `Comments.tex` | Shared LaTeX comment macros used across documents |
| `Common.tex` | Shared LaTeX preamble and common settings |
| `Makefile` | Build script for compiling LaTeX documents |
| `Reflection.tex` | Group reflection document |
| `SRS_Reflection.tex` | SRS-specific reflection |

---

## Building Documents

Each subfolder contains its own `Makefile`. To compile a document, navigate to the subfolder and run:

```bash
make
```

PDFs are already compiled and committed alongside the `.tex` source files for convenience.

---

## Notes

- All documents follow the capstone course template structure
- The SRS uses the Volere template (`SRS-Volere/`)
- `Common.tex` and `Comments.tex` at the root level are shared across multiple documents — do not delete them
