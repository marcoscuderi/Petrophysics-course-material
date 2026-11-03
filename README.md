# Petrophysics course materials

Teaching materials by Marco Scuderi, Sapienza Università di Roma: lecture slides, interactive activities, laboratory protocols, Python tools, notebooks and experimental datasets.

## Lectures

The available lecture folders are numbered consecutively in their original teaching order. This is an index of the supplied materials, not a reconstruction of missing sessions. Original file names and numbers printed within slides are retained for traceability and compatibility.

| Lecture | Topic | Original Drive folder |
|---|---|---|
| 01 | [Course introduction](lectures/01/) | L00 |
| 02 | [Describing rocks quantitatively](lectures/02/) | L01 |
| 03 | [Stress and strain](lectures/03/) | L02 |
| 04 | [Elastic deformation](lectures/04/) | L03 |
| 05 | [Elastic waves and laboratory ultrasonics](lectures/05/) | L04 |
| 06 | [Elastic waves as rock-state probes I](lectures/06/) | L05 |
| 07 | [Elastic waves as rock-state probes II](lectures/07/) | L06 |
| 08 | [UCS data analysis and mechanical–acoustic integration](lectures/08/) | L09 |
| 09 | [Brittle fracture and acoustic emission I](lectures/09/) | L13 |
| 10 | [Brittle fracture and acoustic emission II](lectures/10/) | L14 |
| 11 | [UCS and acoustic-emission laboratory](lectures/11/) | L15_lab |
| 12 | [Acoustic-emission data analysis](lectures/12/) | L16 |

## Using the materials

- **Slides and guides:** open the PDFs in each lecture folder.
- **Interactive activities:** download the HTML file and open it in a browser. Download supporting files together where provided.
- **Laboratory analysis:** retain the supplied directory structure. Input locations may need to be selected or edited for your computer.
- **Experimental data:** smaller datasets are included; large raw AE recordings are linked in [DATA.md](DATA.md). Drive downloads require whatever access the source owner has granted.

## Python tools

A shared starting environment for the supplied tools can be created with Anaconda:

```bash
conda create -n petrophysics python=3.11
conda activate petrophysics
python -m pip install -r requirements.txt
```

Run a GUI from its own directory with `python script_name.py`. Open notebooks with `jupyter lab`. Some tools use Tkinter, which must be installed with your Python distribution. Keep `ae_core.py` alongside the AE analyser.

Dependencies are inferred from the supplied code and are not version-pinned. Python syntax and notebook structure were checked during organization; GUI operation and scientific results have not been revalidated in this environment. Follow the original student guides for detailed workflows and instrument settings.

## Provenance and reuse

Materials were organized from the final supplied Drive folders. The Confalonieri and papers directories, class-group spreadsheet and generated system caches are excluded. Source files are preserved byte-for-byte except five large slide decks, which are split losslessly into ordered PDF parts to fit the upload limit. Complete original PDFs are linked in [DATA.md](DATA.md); [MATERIALS.csv](MATERIALS.csv) records their original labels, repository paths and checksums.

The course draws on *Experimental Rock Deformation — The Brittle Field* by Paterson and Wong. The book and research-paper PDFs are not distributed here. Third-party figures and material retain their original rights and attribution. No additional reuse license has been assigned as part of this transfer.
