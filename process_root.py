import uproot
import awkward as ak # for data manipulation
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt # common shorthand

# Root file location and ldmx-sw pass name
root_file =uproot.open('/Users/mghrear/data/LDMX_GNN/4e_run1.root')['LDMX_Events']
passName="sim" # "TrackerReco"


# Laod in data arrays from root file

# Load Tagger info
Tagger_SimHits_x = root_file['TaggerSimHits_'+passName+'/TaggerSimHits_'+passName+'.x_'].array()
Tagger_SimHits_y = root_file['TaggerSimHits_'+passName+'/TaggerSimHits_'+passName+'.y_'].array()
Tagger_SimHits_z = root_file['TaggerSimHits_'+passName+'/TaggerSimHits_'+passName+'.z_'].array()
Tagger_SimHits_pdgID = root_file['TaggerSimHits_'+passName+'/TaggerSimHits_'+passName+'.pdgID_'].array()
Tagger_SimHits_Edep = root_file['TaggerSimHits_'+passName+'/TaggerSimHits_'+passName+'.edep_'].array()
Tagger_SimHits_trackID = root_file['TaggerSimHits_'+passName+'/TaggerSimHits_'+passName+'.trackID_'].array()

Tagger_Digi_x = root_file['DigiTaggerSimHits_'+passName+'/DigiTaggerSimHits_'+passName+'.x_'].array()
Tagger_Digi_y = root_file['DigiTaggerSimHits_'+passName+'/DigiTaggerSimHits_'+passName+'.y_'].array()
Tagger_Digi_z = root_file['DigiTaggerSimHits_'+passName+'/DigiTaggerSimHits_'+passName+'.z_'].array()
Tagger_Digi_pdgID = root_file['DigiTaggerSimHits_'+passName+'/DigiTaggerSimHits_'+passName+'.pdg_id_'].array()
Tagger_Digi_Edep = root_file['DigiTaggerSimHits_'+passName+'/DigiTaggerSimHits_'+passName+'.edep_'].array() 
Tagger_Digi_trackID = root_file['DigiTaggerSimHits_'+passName+'/DigiTaggerSimHits_'+passName+'.trackIds_'].array()

# Load Recoil info
Recoil_SimHits_x = root_file['RecoilSimHits_'+passName+'/RecoilSimHits_'+passName+'.x_'].array()
Recoil_SimHits_y = root_file['RecoilSimHits_'+passName+'/RecoilSimHits_'+passName+'.y_'].array()
Recoil_SimHits_z = root_file['RecoilSimHits_'+passName+'/RecoilSimHits_'+passName+'.z_'].array()
Recoil_SimHits_pdgID = root_file['RecoilSimHits_'+passName+'/RecoilSimHits_'+passName+'.pdgID_'].array()
Recoil_SimHits_Edep = root_file['RecoilSimHits_'+passName+'/RecoilSimHits_'+passName+'.edep_'].array()
Recoil_SimHits_trackID = root_file['RecoilSimHits_'+passName+'/RecoilSimHits_'+passName+'.trackID_'].array()

Recoil_Digi_x = root_file['DigiRecoilSimHits_'+passName+'/DigiRecoilSimHits_'+passName+'.x_'].array()
Recoil_Digi_y = root_file['DigiRecoilSimHits_'+passName+'/DigiRecoilSimHits_'+passName+'.y_'].array()
Recoil_Digi_z = root_file['DigiRecoilSimHits_'+passName+'/DigiRecoilSimHits_'+passName+'.z_'].array()
Recoil_Digi_pdgID = root_file['DigiRecoilSimHits_'+passName+'/DigiRecoilSimHits_'+passName+'.pdg_id_'].array()
Recoil_Digi_Edep = root_file['DigiRecoilSimHits_'+passName+'/DigiRecoilSimHits_'+passName+'.edep_'].array() 
Recoil_Digi_trackID = root_file['DigiRecoilSimHits_'+passName+'/DigiRecoilSimHits_'+passName+'.trackIds_'].array()


rows = []
for i in range(len(Tagger_SimHits_x)):
    rows.append({"Tagger_Digi_x": np.array(Tagger_Digi_x[i]), "Tagger_Digi_y": np.array(Tagger_Digi_y[i]), "Tagger_Digi_z": np.array(Tagger_Digi_z[i]), "Tagger_Digi_Edep": np.array(Tagger_Digi_Edep[i]), "Tagger_Digi_trackID": np.array(Tagger_Digi_trackID[i]).flatten(), "Tagger_Digi_pdgID": np.array(Tagger_Digi_pdgID[i]), "Recoil_Digi_x": np.array(Recoil_Digi_x[i]), "Recoil_Digi_y": np.array(Recoil_Digi_y[i]), "Recoil_Digi_z": np.array(Recoil_Digi_z[i]), "Recoil_Digi_Edep": np.array(Recoil_Digi_Edep[i]), "Recoil_Digi_trackID": np.array(Recoil_Digi_trackID[i]).flatten(), "Recoil_Digi_pdgID": np.array(Recoil_Digi_pdgID[i])})

df = pd.DataFrame(rows)
df.to_pickle("/Users/mghrear/data/LDMX_GNN/4e_run1.pkl")