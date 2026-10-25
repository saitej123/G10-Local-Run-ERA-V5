Name: Sai Teja
Axiom email: macharlasaiteja@gmail.com
Pack: code-python

### Summary of `code-python` Diagnosis (`bigcode/starcoderdata`)
- **Labelled rows (`labels.csv`):** 180 issue-only documents across 11 disease codes (`A1`, `A2`, `A3`, `A4`, `B1`, `B3`, `C2`, `E1`, `E2`, `E3`, `F1`), formatted per the unified TSAI Lab label specification (`;` without spaces in `diseases`, `|`-separated `evidence` and `hex` in matching order, common control names, `(Nx)` repeat counts, `+`-joined mojibake sequences, and empty `hex` for `F1`/`E3`).
- **Character & control rules (`diagnosis/tools/view_pack.py`):** Adopts the exact `VISIBLE` map (`⟦ZWSP⟧`, `⟦ZWNJ⟧`, `⟦ZWJ⟧`, `⟦BOM⟧`, `⟦SHY⟧`, `⟦LRE⟧..⟦RLO⟧`, `⟦LRI⟧..⟦PDI⟧`, `⟦U+FFFD⟧`) and `Cc` control-character rules (`⟦U+XXXX⟧`).
- **Medicine sheet (`medicine.md`):** 9 medicines covering every disease found more than 5 times (`A1` 45, `F1` 35, `B3` 24, `B1` 19, `C2` 19, `E1` 15, `A2` 11, `A4` 8, `E3` 7), each harm-tested on 200 clean Python documents (`0` changed, `0` dropped; `0.0% <= 1%`).
- **Privacy & safety:** For `F1` (personal data) and `E3` (secrets), only the type and character offset are recorded; no secret or personal values are included.
