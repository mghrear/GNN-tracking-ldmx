import os,math
import argparse
import sys


from LDMX.Framework import ldmxcfg
passName="sim"
p = ldmxcfg.Process(passName)

#import all processors
from LDMX.SimCore import generators
from LDMX.SimCore import simulator
from LDMX.Biasing import filters

from LDMX.Detectors.makePath import *
from LDMX.SimCore import simcfg

#pull in command line options
nEle=int(sys.argv[1])      # simulated beam electrons
runNum=int(sys.argv[2])     # Acts as a seed
version= 'ldmx-det-v14-8gev-no-cals'
outputNameString= str(sys.argv[3]) #sample identifier
outDir= str(sys.argv[4])    #sample identifier

# Instantiate the simulator.
sim = simulator.simulator("test")

# Set the path to the detector to use (pulled from job config)
sim.setDetector( version, True )
sim.scoringPlanes = makeScoringPlanesPath(version)

outname=outDir+"/"+outputNameString+".root"
print("NAME = " + outname)

p.run = runNum
p.maxEvents = 30000
nElectrons = nEle
beamEnergy = 8.0

sim.description = "Inclusive "+str(beamEnergy)+" GeV electron events, "+str(nElectrons)+"e"
sim.beamSpotSmear = [20., 80., 0]


mpgGen = generators.multi( "mgpGen" ) # this is the line that actually creates the generator                                                                  
mpgGen.nParticles = nElectrons
mpgGen.pdgID = 11
mpgGen.enablePoisson = False #True                                                                                                                                      

position =  [ -299.2386690686212, 0.0, -6000.0 ]
momentum =  [ 434.59663056485   , 0.0, 7988.698356992288]
mpgGen.vertex = position
mpgGen.momentum = momentum

# Set the multiparticle gun as generator
sim.generators = [ mpgGen ]

p.sequence = [sim]


# Load the tracking module
from LDMX.Tracking import tracking
# This has to stay after defining the "TrackerReco" Process in order to load the geometry
# From the conditions
from LDMX.Tracking import geo


# Truth seeder 
# Runs truth tracking producing tracks from target scoring plane hits for Recoil
# and generated electros for Tagger.
# Truth tracks can be used for assessing tracking performance or using as seeds

truth_tracking           = tracking.TruthSeedProcessor()
truth_tracking.debug             = False
truth_tracking.pdgIDs            = [11]
truth_tracking.scoring_hits      = "TargetScoringPlaneHits"
truth_tracking.z_min             = 0.
truth_tracking.track_id          = -1
truth_tracking.p_cut             = 0.05 # In MeV
truth_tracking.pz_cut            = 0.03
truth_tracking.p_cutEcal         = 0. # In MeV


# These smearing quantities are default. We expect around 6um hit resolution in bending plane
# v-smearing is actually not used as 1D measurements are used for tracking. These smearing parameters
# are fed to the digitization producer.
uSmearing = 0.006       #mm
vSmearing = 0.000001    #mm


# Smearing Processor - Tagger
# Runs G4 hit smearing producing measurements in the Tagger tracker.
# Hits that belong to the same sensor with the same trackID are merged together to reduce combinatorics
digiTagger = tracking.DigitizationProcessor("DigitizationProcessor")
digiTagger.hit_collection = "TaggerSimHits"
digiTagger.out_collection = "DigiTaggerSimHits"
digiTagger.merge_hits = True
digiTagger.sigma_u = uSmearing
digiTagger.sigma_v = vSmearing

# Smearing Processor - Recoil
digiRecoil = tracking.DigitizationProcessor("DigitizationProcessorRecoil")
digiRecoil.hit_collection = "RecoilSimHits"
digiRecoil.out_collection = "DigiRecoilSimHits"
digiRecoil.merge_hits = True
digiRecoil.sigma_u = uSmearing
digiRecoil.sigma_v = vSmearing

# Seed Finder Tagger
# This runs the track seed finder looking for 5 hits in consecutive sensors and fitting them with a
# parabola+linear fit. Compatibility with expected particles is checked by looking at the track
# parameters and the impact parameters at the target or generation point. For the tagger one should look
# for compatibility with the beam orbit / beam spot
seederTagger = tracking.SeedFinderProcessor("SeedTagger")
seederTagger.input_hits_collection =  digiTagger.out_collection
#seederTagger.perigee_location = [0.,0.,0.]
seederTagger.out_seed_collection = "TaggerRecoSeeds"
seederTagger.pmin  = 0.02743004128947902
seederTagger.pmax  =  63.03941963609926 #4
seederTagger.d0min =  -36.90755883944579 #-60. #-0.5
seederTagger.d0max = 31.512252170025366 #60. #0.5
seederTagger.z0max = 54.622547563516235 #60. #10
seederTagger.thetacut = 0.2622650393957737
seederTagger.phicut =  0.8370135051959274

#Seed finder processor - Recoil
seederRecoil = tracking.SeedFinderProcessor("SeedRecoil")
seederRecoil.perigee_location = [0.,0.,0.]
seederRecoil.input_hits_collection =  digiRecoil.out_collection
seederRecoil.out_seed_collection = "RecoilRecoSeeds"
seederRecoil.pmin  =   0.04311669558561525
seederRecoil.pmax  =  819.0346117063144
seederRecoil.d0min =  -40.20866044915365
seederRecoil.d0max = 36.47833845632701
seederRecoil.z0max = 40.517081118987285
seederRecoil.thetacut =  1.4783718595311992 
seederRecoil.phicut =  1.5786029498528924 



# Producer for running the CKF track finding starting from the found seeds.
tracking_tagger  = tracking.CKFProcessor("Tagger_TrackFinder")
tracking_tagger.dumpobj = False
tracking_tagger.debug = False
tracking_tagger.propagator_step_size = 1000.  #mm
tracking_tagger.const_b_field = False
tracking_tagger.seed_coll_name = "TaggerRecoSeeds" #seederTagger.out_seed_collection #"TaggerTruthSeeds" #
tracking_tagger.out_trk_collection = "TaggerTracks"
tracking_tagger.measurement_collection = digiTagger.out_collection
tracking_tagger.min_hits = 5
tracking_tagger.outlier_pval_ = 16.501226781496662

#CKF Options
tracking_recoil  = tracking.CKFProcessor("Recoil_TrackFinder")
tracking_recoil.dumpobj = False
tracking_recoil.debug = False
tracking_recoil.propagator_step_size = 1000.  #mm
tracking_recoil.bfield = -1.5  #in T #From looking at the BField map
tracking_recoil.const_b_field = False
tracking_recoil.taggerTracking = False
tracking_recoil.seed_coll_name = "RecoilRecoSeeds"
tracking_recoil.out_trk_collection = "RecoilTracks"
tracking_recoil.measurement_collection = digiRecoil.out_collection
tracking_recoil.min_hits = 8
tracking_recoil.outlier_pval_ =  22.165497985508754


GSF_tagger = tracking.GSFProcessor("Tagger_GSF")
GSF_tagger.trackCollection = "TaggerTracksClean"
GSF_tagger.measCollection  = "DigiTaggerSimHits"
GSF_tagger.out_trk_collection = "GSFTagger"
GSF_tagger.taggerTracking = True
GSF_tagger.debug = False

GSF_recoil = tracking.GSFProcessor("Recoil_GSF")
GSF_recoil.trackCollection = "RecoilTracksClean"
GSF_recoil.measCollection  = "DigiRecoilSimHits"
GSF_recoil.out_trk_collection = "GSFRecoil"
GSF_recoil.taggerTracking = False
GSF_recoil.debug = True

greedy_solver_tagger = tracking.GreedyAmbiguitySolver("GreedySolverTagger")
greedy_solver_tagger.nMeasurementsMin = 5
greedy_solver_tagger.maximumSharedHits = 2
greedy_solver_tagger.out_trk_collection = "TaggerTracksClean"
greedy_solver_tagger.trackCollection = "TaggerTracks"
greedy_solver_tagger.measCollection = "DigiTaggerSimHits"

greedy_solver_recoil = tracking.GreedyAmbiguitySolver("GreedySolverRecoil")
greedy_solver_recoil.nMeasurementsMin = 5
greedy_solver_recoil.maximumSharedHits = 2
greedy_solver_recoil.out_trk_collection = "RecoilTracksClean"
greedy_solver_recoil.trackCollection = "RecoilTracks"
greedy_solver_recoil.measCollection = "DigiRecoilSimHits"

from LDMX.Tracking import dqm
digi_dqm = dqm.TrackerDigiDQM()
tracking_dqm = dqm.TrackingRecoDQM()

seed_recoil_dqm = dqm.TrackingRecoDQM("SeedRecoilDQM")
seed_recoil_dqm.track_collection = seederRecoil.out_seed_collection
seed_recoil_dqm.truth_collection = "RecoilTruthSeeds"
seed_recoil_dqm.title = ""
seed_recoil_dqm.buildHistograms()


recoil_dqm = dqm.TrackingRecoDQM("RecoilDQM")
recoil_dqm.track_collection = "GSFRecoil"
recoil_dqm.truth_collection = "RecoilTruthTracks"
recoil_dqm.title = ""
recoil_dqm.buildHistograms()

seed_tagger_dqm = dqm.TrackingRecoDQM("SeedTaggerDQM")
seed_tagger_dqm.track_collection = seederTagger.out_seed_collection
seed_tagger_dqm.truth_collection = "TaggerTruthSeeds"
seed_tagger_dqm.title = ""
seed_tagger_dqm.buildHistograms()


tagger_dqm = dqm.TrackingRecoDQM("TaggerDQM")
tagger_dqm.track_collection = tracking_tagger.out_trk_collection
tagger_dqm.truth_collection = "TaggerTruthTracks"
tagger_dqm.title = ""
tagger_dqm.buildHistograms()


sequence   = [digiTagger, digiRecoil,
                truth_tracking,
                seederTagger, seederRecoil,
                tracking_tagger, tracking_recoil, greedy_solver_tagger, greedy_solver_recoil,
                recoil_dqm, seed_recoil_dqm, seed_tagger_dqm, tagger_dqm]



p.sequence.extend(sequence)









p.termLogLevel=2
p.logFrequency = 1
p.outputFiles = [outname]
p.histogramFile = outDir+'_hists/'+outputNameString