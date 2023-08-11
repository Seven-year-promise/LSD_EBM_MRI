import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

run_name = "reconstrcution_metrics"
main_path="./comp_results/" + run_name + "/" 

out_f = open(main_path + "metrics.txt",'w')

lsd_dice_recon_l_pd = pd.read_csv(os.path.join(main_path, 'lsd_ebm_dice_l_recon.csv'), sep='\t')
print("-------------------------lsd_dice_recon_l_pd", file=out_f)
print(lsd_dice_recon_l_pd.mean(), lsd_dice_recon_l_pd.std(), file=out_f)
lsd_dice_recon_h_pd = pd.read_csv(os.path.join(main_path, 'lsd_ebm_dice_h_recon.csv'), sep='\t')
print("-------------------------lsd_dice_recon_h_pd", file=out_f)
print(lsd_dice_recon_h_pd.mean(), lsd_dice_recon_h_pd.std(), file=out_f)
lsd_vs_recon_l_pd = pd.read_csv(os.path.join(main_path, 'lsd_ebm_vs_l_recon.csv'), sep='\t')
print("-------------------------lsd_vs_recon_l_pd", file=out_f)
print(lsd_vs_recon_l_pd.mean(), lsd_vs_recon_l_pd.std(), file=out_f)
lsd_vs_recon_h_pd = pd.read_csv(os.path.join(main_path, 'lsd_ebm_vs_h_recon.csv'), sep='\t')
print("-------------------------lsd_vs_recon_h_pd", file=out_f)
print(lsd_vs_recon_h_pd.mean(), lsd_vs_recon_h_pd.std(), file=out_f)
lsd_hd_recon_l_pd = pd.read_csv(os.path.join(main_path, 'lsd_ebm_hd_l_recon.csv'), sep='\t')
print("-------------------------lsd_hd_recon_l_pd", file=out_f)
print(lsd_hd_recon_l_pd.mean(), lsd_hd_recon_l_pd.std(), file=out_f)
lsd_hd_recon_h_pd = pd.read_csv(os.path.join(main_path, 'lsd_ebm_hd_h_recon.csv'), sep='\t')
print("-------------------------lsd_hd_recon_h_pd", file=out_f)
print(lsd_hd_recon_h_pd.mean(), lsd_hd_recon_h_pd.std(), file=out_f)
lsd_sen_recon_l_pd = pd.read_csv(os.path.join(main_path, 'lsd_ebm_sen_l_recon.csv'), sep='\t')
print("-------------------------lsd_sen_recon_l_pd", file=out_f)
print(lsd_sen_recon_l_pd.mean(), lsd_sen_recon_l_pd.std(), file=out_f)
lsd_sen_recon_h_pd = pd.read_csv(os.path.join(main_path, 'lsd_ebm_sen_h_recon.csv'), sep='\t')
print("-------------------------lsd_sen_recon_h_pd", file=out_f)
print(lsd_sen_recon_h_pd.mean(), lsd_sen_recon_h_pd.std(), file=out_f)
lsd_spec_recon_l_pd = pd.read_csv(os.path.join(main_path, 'lsd_ebm_spec_l_recon.csv'), sep='\t')
print("-------------------------lsd_spec_recon_l_pd", file=out_f)
print(lsd_spec_recon_l_pd.mean(), lsd_spec_recon_l_pd.std(), file=out_f)
lsd_spec_recon_h_pd = pd.read_csv(os.path.join(main_path, 'lsd_ebm_spec_h_recon.csv'), sep='\t')
print("-------------------------lsd_spec_recon_h_pd", file=out_f)
print(lsd_spec_recon_h_pd.mean(), lsd_spec_recon_h_pd.std(), file=out_f)


lsd_nmi_recon_l_pd = pd.read_csv(os.path.join(main_path, 'lsd_ebm_nmi_l_recon.csv'), sep='\t')
print("-------------------------lsd_nmi_recon_l_pd", file=out_f)
print(lsd_nmi_recon_l_pd.mean(), lsd_nmi_recon_l_pd.std(), file=out_f)
lsd_nmi_recon_h_pd = pd.read_csv(os.path.join(main_path, 'lsd_ebm_nmi_h_recon.csv'), sep='\t')
print("-------------------------lsd_nmi_recon_h_pd", file=out_f)
print(lsd_nmi_recon_h_pd.mean(), lsd_nmi_recon_h_pd.std(), file=out_f)

lsd_ck_recon_l_pd = pd.read_csv(os.path.join(main_path, 'lsd_ebm_ck_l_recon.csv'), sep='\t')
print("-------------------------lsd_ck_recon_l_pd", file=out_f)
print(lsd_ck_recon_l_pd.mean(), lsd_ck_recon_l_pd.std(), file=out_f)
lsd_ck_recon_h_pd = pd.read_csv(os.path.join(main_path, 'lsd_ebm_ck_h_recon.csv'), sep='\t')
print("-------------------------lsd_ck_recon_h_pd", file=out_f)
print(lsd_ck_recon_h_pd.mean(), lsd_ck_recon_h_pd.std(), file=out_f)

#lebm
lebm_dice_recon_l_pd = pd.read_csv(os.path.join(main_path, 'lebm_dice_l_recon.csv'), sep='\t')
print("-------------------------lebm_dice_recon_l_pd", file=out_f)
print(lebm_dice_recon_l_pd.mean(), lebm_dice_recon_l_pd.std(), file=out_f)
lebm_dice_recon_h_pd = pd.read_csv(os.path.join(main_path, 'lebm_dice_h_recon.csv'), sep='\t')
print("-------------------------lebm_dice_recon_h_pd", file=out_f)
print(lebm_dice_recon_h_pd.mean(), lebm_dice_recon_h_pd.std(), file=out_f)
lebm_vs_recon_l_pd = pd.read_csv(os.path.join(main_path, 'lebm_vs_l_recon.csv'), sep='\t')
print("-------------------------lebm_vs_recon_l_pd", file=out_f)
print(lebm_vs_recon_l_pd.mean(), lebm_vs_recon_l_pd.std(), file=out_f)
lebm_vs_recon_h_pd = pd.read_csv(os.path.join(main_path, 'lebm_vs_h_recon.csv'), sep='\t')
print("-------------------------lebm_vs_recon_h_pd", file=out_f)
print(lebm_vs_recon_h_pd.mean(), lebm_vs_recon_h_pd.std(), file=out_f)
lebm_hd_recon_l_pd = pd.read_csv(os.path.join(main_path, 'lebm_hd_l_recon.csv'), sep='\t')
print("-------------------------lebm_hd_recon_l_pd", file=out_f)
print(lebm_hd_recon_l_pd.mean(), lebm_hd_recon_l_pd.std(), file=out_f)
lebm_hd_recon_h_pd = pd.read_csv(os.path.join(main_path, 'lebm_hd_h_recon.csv'), sep='\t')
print("-------------------------lebm_hd_recon_h_pd", file=out_f)
print(lebm_hd_recon_h_pd.mean(), lebm_hd_recon_h_pd.std(), file=out_f)
lebm_sen_recon_l_pd = pd.read_csv(os.path.join(main_path, 'lebm_sen_l_recon.csv'), sep='\t')
print("-------------------------lebm_sen_recon_l_pd", file=out_f)
print(lebm_sen_recon_l_pd.mean(), lebm_sen_recon_l_pd.std(), file=out_f)
lebm_sen_recon_h_pd = pd.read_csv(os.path.join(main_path, 'lebm_sen_h_recon.csv'), sep='\t')
print("-------------------------lebm_sen_recon_h_pd", file=out_f)
print(lebm_sen_recon_h_pd.mean(), lebm_sen_recon_h_pd.std(), file=out_f)
lebm_spec_recon_l_pd = pd.read_csv(os.path.join(main_path, 'lebm_spec_l_recon.csv'), sep='\t')
print("-------------------------lebm_spec_recon_l_pd", file=out_f)
print(lebm_spec_recon_l_pd.mean(), lebm_spec_recon_l_pd.std(), file=out_f)
lebm_spec_recon_h_pd = pd.read_csv(os.path.join(main_path, 'lebm_spec_h_recon.csv'), sep='\t')
print("-------------------------lebm_spec_recon_h_pd", file=out_f)
print(lebm_spec_recon_h_pd.mean(), lebm_spec_recon_h_pd.std(), file=out_f)

lebm_nmi_recon_l_pd = pd.read_csv(os.path.join(main_path, 'lebm_nmi_l_recon.csv'), sep='\t')
print("-------------------------lebm_nmi_recon_l_pd", file=out_f)
print(lebm_nmi_recon_l_pd.mean(), lebm_nmi_recon_l_pd.std(), file=out_f)
lebm_nmi_recon_h_pd = pd.read_csv(os.path.join(main_path, 'lebm_nmi_h_recon.csv'), sep='\t')
print("-------------------------lebm_nmi_recon_h_pd", file=out_f)
print(lebm_nmi_recon_h_pd.mean(), lebm_nmi_recon_h_pd.std(), file=out_f)

lebm_ck_recon_l_pd = pd.read_csv(os.path.join(main_path, 'lebm_ck_l_recon.csv'), sep='\t')
print("-------------------------lebm_ck_recon_l_pd", file=out_f)
print(lebm_ck_recon_l_pd.mean(), lebm_ck_recon_l_pd.std(), file=out_f)
lebm_ck_recon_h_pd = pd.read_csv(os.path.join(main_path, 'lebm_ck_h_recon.csv'), sep='\t')
print("-------------------------lebm_ck_recon_h_pd", file=out_f)
print(lebm_ck_recon_h_pd.mean(), lebm_ck_recon_h_pd.std(), file=out_f)


# vae
vae_dice_recon_l_pd = pd.read_csv(os.path.join(main_path, 'vae_dice_l_recon.csv'), sep='\t')
print("-------------------------vae_dice_recon_l_pd", file=out_f)
print(vae_dice_recon_l_pd.mean(), vae_dice_recon_l_pd.std(), file=out_f)
vae_dice_recon_h_pd = pd.read_csv(os.path.join(main_path, 'vae_dice_h_recon.csv'), sep='\t')
print("-------------------------vae_dice_recon_h_pd", file=out_f)
print(vae_dice_recon_h_pd.mean(), vae_dice_recon_h_pd.std(), file=out_f)
vae_vs_recon_l_pd = pd.read_csv(os.path.join(main_path, 'vae_vs_l_recon.csv'), sep='\t')
print("-------------------------vae_vs_recon_l_pd", file=out_f)
print(vae_vs_recon_l_pd.mean(), vae_vs_recon_l_pd.std(), file=out_f)
vae_vs_recon_h_pd = pd.read_csv(os.path.join(main_path, 'vae_vs_h_recon.csv'), sep='\t')
print("-------------------------vae_vs_recon_h_pd", file=out_f)
print(vae_vs_recon_h_pd.mean(), vae_vs_recon_h_pd.std(), file=out_f)
vae_hd_recon_l_pd = pd.read_csv(os.path.join(main_path, 'vae_hd_l_recon.csv'), sep='\t')
print("-------------------------vae_hd_recon_l_pd", file=out_f)
print(vae_hd_recon_l_pd.mean(), vae_hd_recon_l_pd.std(), file=out_f)
vae_hd_recon_h_pd = pd.read_csv(os.path.join(main_path, 'vae_hd_h_recon.csv'), sep='\t')
print("-------------------------vae_hd_recon_h_pd", file=out_f)
print(vae_hd_recon_h_pd.mean(), vae_hd_recon_h_pd.std(), file=out_f)
vae_sen_recon_l_pd = pd.read_csv(os.path.join(main_path, 'vae_sen_l_recon.csv'), sep='\t')
print("-------------------------vae_sen_recon_l_pd", file=out_f)
print(vae_sen_recon_l_pd.mean(), vae_sen_recon_l_pd.std(), file=out_f)
vae_sen_recon_h_pd = pd.read_csv(os.path.join(main_path, 'vae_sen_h_recon.csv'), sep='\t')
print("-------------------------vae_sen_recon_h_pd", file=out_f)
print(vae_sen_recon_h_pd.mean(), vae_sen_recon_h_pd.std(), file=out_f)
vae_spec_recon_l_pd = pd.read_csv(os.path.join(main_path, 'vae_spec_l_recon.csv'), sep='\t')
print("-------------------------vae_spec_recon_l_pd", file=out_f)
print(vae_spec_recon_l_pd.mean(), vae_spec_recon_l_pd.std(), file=out_f)
vae_spec_recon_h_pd = pd.read_csv(os.path.join(main_path, 'vae_spec_h_recon.csv'), sep='\t')
print("-------------------------vae_spec_recon_h_pd", file=out_f)
print(vae_spec_recon_h_pd.mean(), vae_spec_recon_h_pd.std(), file=out_f)

vae_nmi_recon_l_pd = pd.read_csv(os.path.join(main_path, 'vae_nmi_l_recon.csv'), sep='\t')
print("-------------------------vae_nmi_recon_l_pd", file=out_f)
print(vae_nmi_recon_l_pd.mean(), vae_nmi_recon_l_pd.std(), file=out_f)
vae_nmi_recon_h_pd = pd.read_csv(os.path.join(main_path, 'vae_nmi_h_recon.csv'), sep='\t')
print("-------------------------vae_nmi_recon_h_pd", file=out_f)
print(vae_nmi_recon_h_pd.mean(), vae_nmi_recon_h_pd.std(), file=out_f)

vae_ck_recon_l_pd = pd.read_csv(os.path.join(main_path, 'vae_ck_l_recon.csv'), sep='\t')
print("-------------------------vae_ck_recon_l_pd", file=out_f)
print(vae_ck_recon_l_pd.mean(), vae_ck_recon_l_pd.std(), file=out_f)
vae_ck_recon_h_pd = pd.read_csv(os.path.join(main_path, 'vae_ck_h_recon.csv'), sep='\t')
print("-------------------------vae_ck_recon_h_pd", file=out_f)
print(vae_ck_recon_h_pd.mean(), vae_ck_recon_h_pd.std(), file=out_f)
"""
boxes = plt.boxplot([lsd_dice_recon_l_pd["2"].tolist(), lebm_dice_recon_l_pd["2"].tolist(),
                     lsd_dice_recon_l_pd["15"].tolist(), lebm_dice_recon_l_pd["15"].tolist(),
                     lsd_dice_recon_l_pd["20"].tolist(), lebm_dice_recon_l_pd["20"].tolist()],
                     labels=['2', "", '15', "", '20', ""], widths=0.5, positions=[1,2,4,5,7,8], patch_artist=True,
                     showfliers=True, showmeans=True)
#for box, color in zip(boxes['boxes'], colors):
#    box.set(facecolor=color)
plt.xticks(fontsize=14, fontname = "Times New Roman")
plt.yticks(fontsize=14, fontname = "Times New Roman")
#plt.title(ylabels[3], fontname = "Times New Roman", fontsize=14)

#ax = lsd_dice_recon_l_pd[["2", "15", "20"]].plot(kind='box', title='boxplot')

#ax = lsd_dice_recon_h_pd[["2", "15", "20"]].plot(kind='box', title='boxplot')
# Display the plot
plt.show()

#plt.savefig(os.path.join(main_path, "lsd_dice_recon_l.png"))

#KEY = "DICE_L"
#KEY = "DICE_H"
#KEY = "VS_L"
#KEY = "VS_H"
#KEY = "SEN_L"
#KEY = "SEN_H"
#KEY = "SPEC_L"
KEY = "SPEC_H"

data = {
    "DICE_L": [lsd_dice_recon_l_pd, lebm_dice_recon_l_pd],
    "DICE_H": [lsd_dice_recon_h_pd, lebm_dice_recon_h_pd],
    "VS_L": [lsd_vs_recon_l_pd, lebm_vs_recon_l_pd],
    "VS_H": [lsd_vs_recon_h_pd, lebm_vs_recon_h_pd],
    "SEN_L": [lsd_sen_recon_l_pd, lebm_sen_recon_l_pd],
    "SEN_H": [lsd_sen_recon_h_pd, lebm_sen_recon_h_pd],
    "SPEC_L": [lsd_spec_recon_l_pd, lebm_spec_recon_l_pd],
    "SPEC_H": [lsd_spec_recon_h_pd, lebm_spec_recon_h_pd],
}

# the list named ticks, summarizes or groups
# the summer and winter rainfall as low, mid
# and high
ticks = ['2', '15', '20']
y_label = {
    "DICE_L": "DICE (L)",
    "DICE_H": "DICE (H)",
    "VS_L": "VS (L)",
    "VS_H": "VS (H)",
    "SEN_L": "Sensibility (L)",
    "SEN_H": "Sensibility (H)",
    "SPEC_L": "Specificity (L)",
    "SPEC_H": "Specificity (H)"
}
save_path = {
    "DICE_L": os.path.join(main_path, "dice_l.eps"),
    "DICE_H": os.path.join(main_path, "dice_h.eps"),
    "VS_L": os.path.join(main_path, "vs_l.eps"),
    "VS_H": os.path.join(main_path, "vs_h.eps"),
    "SEN_L": os.path.join(main_path, "sen_l.eps"),
    "SEN_H": os.path.join(main_path, "sen_h.eps"),
    "SPEC_L": os.path.join(main_path, "spec_l.eps"),
    "SPEC_H": os.path.join(main_path, "spec_h.eps")
}
# create a boxplot for two arrays separately,
# the position specifies the location of the
# particular box in the graph,
# this can be changed as per your wish. Use width
# to specify the width of the plot
lsd_dice_recon_l_plot = plt.boxplot([list(data[KEY][0]["2"]), list(data[KEY][0]["15"]), list(data[KEY][0]["20"])],
                               positions=np.array(
                                   np.arange(3)) * 2.0 - 0.35,
                               widths=0.6)
lebm_dice_recon_l_plot = plt.boxplot([list(data[KEY][1]["2"]), list(data[KEY][1]["15"]), list(data[KEY][1]["20"])],
                               positions=np.array(
                                   np.arange(3)) * 2.0 + 0.35,
                               widths=0.6)


# each plot returns a dictionary, use plt.setp()
# function to assign the color code
# for all properties of the box plot of particular group
# use the below function to set color for particular group,
# by iterating over all properties of the box plot
def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)

    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend()


# setting colors for each groups
define_box_properties(lsd_dice_recon_l_plot, '#D7191C', 'LSD-EBM')
define_box_properties(lebm_dice_recon_l_plot, '#2C7BB6', 'LEBM')

# set the x label values
plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks, fontname="Arial", fontsize=16)
plt.yticks(fontname="Arial", fontsize=16)
# set the limit for x axis
#plt.xlim(-2, len(ticks) * 2)
plt.xlabel("Steps (Diffusion or MCMC)", fontname="Arial", fontsize=16)
plt.ylabel(y_label[KEY], fontname="Arial", fontsize=16)
# set the limit for y axis
#plt.ylim(0, 50)

# set the title
plt.legend(prop={'family':"Arial", 'size':16})
plt.tight_layout()
plt.savefig(save_path[KEY])
"""