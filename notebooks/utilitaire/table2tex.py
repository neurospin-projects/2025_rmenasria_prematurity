# save_as_latex_tables.py
# Usage: python save_as_latex_tables.py dhcp_results.csv
import sys
import pandas as pd

fn = sys.argv[1] if len(sys.argv) > 1 else "/neurospin/dico/rmenasria/Runs/03_main/Output/final/classif_prematurity_dHCP_final_28_32.csv"
df = pd.read_csv(fn)

# normalize region -> base name + hemi
def split_region(r):
    if r.endswith("_left"):
        return r[:-5], "left"
    if r.endswith("_right"):
        return r[:-6], "right"
    return r, "center"

df[['region_base','hemi']] = df['region'].apply(lambda x: pd.Series(split_region(x)))

tranches = ['<28','28-32','32-37']

def esc(s):
    # escape underscores for LaTeX
    return s.replace('_', '\\_')

def build_rows_for_hemi(df_hemi):
    rows = []
    for reg in sorted(df_hemi['region_base'].unique()):
        row = [esc(reg)]
        for t in tranches:
            sel = df_hemi[(df_hemi['region_base']==reg)]
            if not sel.empty:
                s = sel.iloc[0]
                row += [f"{s['AUC_mean']:.6f}", f"{s['AUC_std']:.6f}", f"{s['perm_pval']:.6f}"]
            else:
                row += ['-', '-', '-']
        rows.append(row)
    return rows

def make_table_string(rows, caption, label):
    header = "\\begin{table}[ht]\n\\centering\n\\small\n\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
    header += "Region & AUC $<28$ & STD & p & AUC 28--32 & STD & p & AUC 32--37 & STD & p \\\\\n\\midrule\n"
    body = ""
    for r in rows:
        body += " & ".join(r) + " \\\\\n"
    footer = "\\bottomrule\n\\end{tabular}\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\end{table}\n"
    return header + body + footer

# LEFT hemisphere
df_left = df[df['hemi']=='left']
rows_left = build_rows_for_hemi(df_left)
latex_left = make_table_string(rows_left,
                              "AUC results on dHCP --- left hemisphere. For each gestational age threshold: mean AUC, standard deviation, and permutation p-value.",
                              "tab:auc_dhcp_left")

# RIGHT hemisphere
df_right = df[df['hemi']=='right']
rows_right = build_rows_for_hemi(df_right)
latex_right = make_table_string(rows_right,
                               "AUC results on dHCP --- right hemisphere. For each gestational age threshold: mean AUC, standard deviation, and permutation p-value.",
                               "tab:auc_dhcp_right")

# Save to files
with open("auc_dhcp_left.tex","w",encoding="utf-8") as f:
    f.write(latex_left)
with open("auc_dhcp_right.tex","w",encoding="utf-8") as f:
    f.write(latex_right)

print("Generated files: auc_dhcp_left.tex and auc_dhcp_right.tex")
print("Open them and paste the table contents into your Overleaf .tex file.")
