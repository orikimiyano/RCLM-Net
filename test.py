import os
import warnings
from data import *
from models.RCLM_Net import *


warnings.filterwarnings("ignore")

NameOfModel = 'RCLM_Net'

##############1ST#################
Cross_Validation = '1ST'
train_index = 'Multi_5CR/'+Cross_Validation+'\data/'

def test(case_path_1,case_path_2,case_path_3,case_name):
    testGene = testGenerator(case_path_1,case_path_2,case_path_3)
    lens = getFileNum(case_path_1)
    model = net(num_class=20)
    model.load_weights('saved_models/'+str(NameOfModel)+'_'+str(Cross_Validation)+'.hdf5')
    results = model.predict_generator(testGene,lens,max_queue_size=1,verbose=1)
    if not(os.path.exists('results/'+str(NameOfModel)+'_'+str(Cross_Validation)+'/'+case_name)):
        os.makedirs('results/'+str(NameOfModel)+'_'+str(Cross_Validation)+'/'+case_name)
    saveResult('results/'+str(NameOfModel)+'_'+str(Cross_Validation)+'/'+case_name,results)

case_path_1 = str(train_index) + 'test/T2'
case_path_2 = str(train_index) + 'test/T1'
case_path_3 = str(train_index) + 'test/T2_FS'

print(case_path_1)
print(case_path_2)
print(case_path_3)
# test(case_path_1,case_path_2,case_path_3,case_path_3)

print('1ST test')

for root_1, dirs_1, files_1 in os.walk(case_path_1):
    for dir in dirs_1:
        one_case_root_1 = os.path.join(case_path_1, dir)
        one_case_root_2 = os.path.join(case_path_2, dir)
        one_case_root_3 = os.path.join(case_path_3, dir)
        test(one_case_root_1,one_case_root_2,one_case_root_3, dir)

##############2ND#################
Cross_Validation = '2ND'
train_index = 'Multi_5CR/'+Cross_Validation+'\data/'

def test(case_path_1,case_path_2,case_path_3,case_name):
    testGene = testGenerator(case_path_1,case_path_2,case_path_3)
    lens = getFileNum(case_path_1)
    model = net(num_class=20)
    model.load_weights('saved_models/'+str(NameOfModel)+'_'+str(Cross_Validation)+'.hdf5')
    results = model.predict_generator(testGene,lens,max_queue_size=1,verbose=1)
    if not(os.path.exists('results/'+str(NameOfModel)+'_'+str(Cross_Validation)+'/'+case_name)):
        os.makedirs('results/'+str(NameOfModel)+'_'+str(Cross_Validation)+'/'+case_name)
    saveResult('results/'+str(NameOfModel)+'_'+str(Cross_Validation)+'/'+case_name,results)

case_path_1 = str(train_index) + 'test/T2'
case_path_2 = str(train_index) + 'test/T1'
case_path_3 = str(train_index) + 'test/T2_FS'

# test(case_path_1,case_path_2,case_path_3,case_path_3)
print('2nd test')
for root_1, dirs_1, files_1 in os.walk(case_path_1):
    for dir in dirs_1:
        one_case_root_1 = os.path.join(case_path_1, dir)
        one_case_root_2 = os.path.join(case_path_2, dir)
        one_case_root_3 = os.path.join(case_path_3, dir)
        test(one_case_root_1,one_case_root_2,one_case_root_3, dir)

##############3RD#################
Cross_Validation = '3RD'
train_index = 'Multi_5CR/'+Cross_Validation+'\data/'

def test(case_path_1,case_path_2,case_path_3,case_name):
    testGene = testGenerator(case_path_1,case_path_2,case_path_3)
    lens = getFileNum(case_path_1)
    model = net(num_class=20)
    model.load_weights('saved_models/'+str(NameOfModel)+'_'+str(Cross_Validation)+'.hdf5')
    results = model.predict_generator(testGene,lens,max_queue_size=1,verbose=1)
    if not(os.path.exists('results/'+str(NameOfModel)+'_'+str(Cross_Validation)+'/'+case_name)):
        os.makedirs('results/'+str(NameOfModel)+'_'+str(Cross_Validation)+'/'+case_name)
    saveResult('results/'+str(NameOfModel)+'_'+str(Cross_Validation)+'/'+case_name,results)

case_path_1 = str(train_index) + 'test/T2'
case_path_2 = str(train_index) + 'test/T1'
case_path_3 = str(train_index) + 'test/T2_FS'

# test(case_path_1,case_path_2,case_path_3,case_path_3)
print('3RD test')
for root_1, dirs_1, files_1 in os.walk(case_path_1):
    for dir in dirs_1:
        one_case_root_1 = os.path.join(case_path_1, dir)
        one_case_root_2 = os.path.join(case_path_2, dir)
        one_case_root_3 = os.path.join(case_path_3, dir)
        test(one_case_root_1,one_case_root_2,one_case_root_3, dir)

##############4TH#################
Cross_Validation = '4TH'
train_index = 'Multi_5CR/'+Cross_Validation+'\data/'

def test(case_path_1,case_path_2,case_path_3,case_name):
    testGene = testGenerator(case_path_1,case_path_2,case_path_3)
    lens = getFileNum(case_path_1)
    model = net(num_class=20)
    model.load_weights('saved_models/'+str(NameOfModel)+'_'+str(Cross_Validation)+'.hdf5')
    results = model.predict_generator(testGene,lens,max_queue_size=1,verbose=1)
    if not(os.path.exists('results/'+str(NameOfModel)+'_'+str(Cross_Validation)+'/'+case_name)):
        os.makedirs('results/'+str(NameOfModel)+'_'+str(Cross_Validation)+'/'+case_name)
    saveResult('results/'+str(NameOfModel)+'_'+str(Cross_Validation)+'/'+case_name,results)

case_path_1 = str(train_index) + 'test/T2'
case_path_2 = str(train_index) + 'test/T1'
case_path_3 = str(train_index) + 'test/T2_FS'

# test(case_path_1,case_path_2,case_path_3,case_path_3)
print('4TH test')
for root_1, dirs_1, files_1 in os.walk(case_path_1):
    for dir in dirs_1:
        one_case_root_1 = os.path.join(case_path_1, dir)
        one_case_root_2 = os.path.join(case_path_2, dir)
        one_case_root_3 = os.path.join(case_path_3, dir)
        test(one_case_root_1,one_case_root_2,one_case_root_3, dir)

##############5ST#################
Cross_Validation = '5TH'
train_index = 'Multi_5CR/'+Cross_Validation+'\data/'

def test(case_path_1,case_path_2,case_path_3,case_name):
    testGene = testGenerator(case_path_1,case_path_2,case_path_3)
    lens = getFileNum(case_path_1)
    model = net(num_class=20)
    model.load_weights('saved_models/'+str(NameOfModel)+'_'+str(Cross_Validation)+'.hdf5')
    results = model.predict_generator(testGene,lens,max_queue_size=1,verbose=1)
    if not(os.path.exists('results/'+str(NameOfModel)+'_'+str(Cross_Validation)+'/'+case_name)):
        os.makedirs('results/'+str(NameOfModel)+'_'+str(Cross_Validation)+'/'+case_name)
    saveResult('results/'+str(NameOfModel)+'_'+str(Cross_Validation)+'/'+case_name,results)

case_path_1 = str(train_index) + 'test/T2'
case_path_2 = str(train_index) + 'test/T1'
case_path_3 = str(train_index) + 'test/T2_FS'

# test(case_path_1,case_path_2,case_path_3,case_path_3)
print('5TH test')
for root_1, dirs_1, files_1 in os.walk(case_path_1):
    for dir in dirs_1:
        one_case_root_1 = os.path.join(case_path_1, dir)
        one_case_root_2 = os.path.join(case_path_2, dir)
        one_case_root_3 = os.path.join(case_path_3, dir)
        test(one_case_root_1,one_case_root_2,one_case_root_3, dir)
