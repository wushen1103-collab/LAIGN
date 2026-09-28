# External baselines

These adapters convert outputs from externally maintained methods to the
residue-ranking protocol used by LAIGN. Obtain each method and checkpoint from
its official repository and comply with its license. Third-party source code,
weights and datasets are not redistributed here.

The adapters never provide a baseline with information beyond its native
test-time inputs. Evaluation is performed on matched complex identifiers and
reports coverage when an external method cannot process every complex.
