Reproducibility materials for Competing Rules, Shared Seas

EF-R1/analysis contains the frozen derived inputs, reference and independent results, numerical programs, input hashes, requirements, environment record and license. To reproduce the numerical analyses, install requirements.txt and run audit_activation.py, recompute_rule_comparison.py and run_sensitivity_and_market.py in that order from that directory. make_revision_figures.py produces the model and sensitivity charts in a separate output directory.

EF-Figures-v5/panel_data contains the exact tables used for the geographic and statistical panels. The final main and supporting PDFs are in the submission package. To render the maps from the archived results, keep the directory structure, install the listed Python dependencies plus PyMuPDF and Pillow, and make Times New Roman available. Run python EF-Figures-v5/render_maps.py from the extracted archive root. This writes figures_main, figures_supporting and panel_data under EF-Figures-v5.

Figure S1 uses the accepted coastal rendering supplied in EF-Figures-v4-S1. Run draw_s1.py in that directory only to regenerate the proxy itself. The bathymetric and ocean data are from Natural Earth.

Revised manuscript verification

From the extracted archive root, run:
  python EF-R1/analysis/check_footprint_shares.py --output reproduced_shares.csv

Compare the resulting eight rows with EF-R1/analysis/results/R1_footprint_shares.csv. The script reads the country-code list, candidate ports and governance corridor table from EF-R1/analysis/inputs/data. It first checks all 7,769 reference BRI flags and all four reference labels against the archived route records. It then removes IT, TW and IN, preserving EU endpoint priority and the original proximity flag. The denominator in both scenarios is the annual emissions total of all 7,769 corridors. In EU-connected, BRI-only, near-EU and outside-footprint order, the 56-code scenario has 1,730, 3,908, 1,180 and 951 corridors and accounts for 20.9%, 54.4%, 15.8% and 8.9% of emissions, respectively.

Final figure sources

Figure1/Figure_01.tex is the TikZ source of main Figure 1. Compile from the Figure1 directory with "xelatex Figure_01.tex" and Times New Roman installed.

figures_main/Figure_06.py and figures_main/Figure_08.py generate the final layouts of main Figures 6 and 8. Run the following from the extracted archive root, replacing PATH_TO_ZIP with the path to the downloaded reproducibility ZIP. Figure 6 reads the archived map panels and panel tables with PyMuPDF. Figure 8 reads the archived financing scenarios with Matplotlib. The submitted Figure 8 was rendered with Matplotlib 3.10.0 and Times New Roman. Each command writes a PDF to figures_main.

  python figures_main/Figure_06.py "PATH_TO_ZIP" --output figures_main/Figure_06.pdf
  python figures_main/Figure_08.py --archive "PATH_TO_ZIP" --output figures_main/Figure_08.pdf

The staged investment program for Figure 5 and its input and output descriptions are in Figure5. Instructions for updating the numerical results are in Figure5/README.txt.
