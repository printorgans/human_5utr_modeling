import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn import preprocessing
from keras.models import load_model
import h5py

# import keras
# from keras.preprocessing import sequence
# from keras.optimizers import RMSprop
# from keras.models import Sequential
# from keras.layers.core import Dense
# from keras.layers.core import Dropout
# from keras.layers.core import Activation
# from keras.layers.core import Flatten
# from keras.layers.convolutional import Convolution1D
# from keras.constraints import maxnorm

np.random.seed(1337)

pd.set_option("display.max_colwidth",100)
sns.set(style="ticks", color_codes=True)

def plot_data(data, x, y, x_title='x', y_title='y', xlim=None, ylim=None, size=4, alpha=0.02):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data[x], data[y])
    print(f"# of UTRs: {len(data)}")
    print(f"r-squared: {r_value**2}")

    sns.set(style="ticks", color_codes=True)
    g = sns.JointGrid(data=data, x=x, y=y, xlim=xlim, ylim=ylim, height=size)
    g = g.plot_joint(plt.scatter, color='#e01145', edgecolor="black", alpha=alpha)
    f = g.fig
    f.text(x=0, y=0, s='r2 = {}'.format(round(r_value**2, 3)))
    g = g.plot_marginals(sns.histplot, kde=False, color='#e01145')
    g = g.set_axis_labels(x_title, y_title)
    plt.show()
    

def one_hot_encode(df, col='utr', seq_len=50):
    # Dictionary returning one-hot encoding of nucleotides. 
    nuc_d = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}
    
    # Creat empty matrix.
    vectors=np.empty([len(df),seq_len,4])
    
    # Iterate through UTRs and one-hot encode
    for i,seq in enumerate(df[col].str[:seq_len]): 
        seq = seq.lower()
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors

def test_data(df, model, test_seq, obs_col, output_col='pred'):
    scaler = preprocessing.StandardScaler()
    scaler.fit(df[obs_col].reshape(-1,1))
    #df.loc[:,'obs_stab'] = test_df['stab_df']
    predictions = model.predict(test_seq).reshape(-1)
    df.loc[:,output_col] = scaler.inverse_transform(predictions)
    return df

def binarize_sequences(df, col='utr', seq_len=54):
    vector=np.empty([len(df),seq_len,4])
    for i,seq in enumerate(df[col].str[:seq_len]):
        vector[i]=vectorizeSequence(seq.lower())
    return vector

def vectorizeSequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    ltrdict = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}
    return np.array([ltrdict[x] for x in seq])


df = pd.read_csv('./data/GSM3130435_egfp_unmod_1.csv.gz')
df.sort_values('total_reads', ascending=False).reset_index(drop=True)

# Select a number of UTRs for the purpose of scaling.
scale_utrs = df[:40000]

# Scale
scaler = preprocessing.StandardScaler()
# scaler.fit(scale_utrs['rl'].reshape(-1,1))
scaler.fit(scale_utrs['rl'].values.reshape(-1, 1))


# model = load_model('../modeling/saved_models/evolution_model.hdf5')
# model = load_model('../modeling/saved_models/retrained_evolution_model.hdf5')
# model = load_model('../modeling/saved_models/main_MRL_model.hdf5')




#plot_data(df, 'rl', 'total_reads', x_title='Read Length', y_title='Total Reads', xlim=(0, 100), ylim=(0, 1000), size=6, alpha=0.1)


# model = load_model('./modeling/saved_models/retrained_main_MRL_model.h5')
# Load the model
try:
    model = load_model('./modeling/saved_models/retrained_evolution_model.hdf5')
except TypeError as e:
    print(f"Error occurred: {e}")
    with h5py.File('./modeling/saved_models/retrained_evolution_model.hdf5', 'r') as f:
        config = f.attrs.get('model_config')
        print(f"model_config: {config}")
        # Attempt to decode the config if it's in bytes
        if isinstance(config, bytes):
            config = config.decode('utf-8')
        print(f"Decoded model_config: {config}")
        # Parse the JSON configuration to inspect it
        import json
        config_dict = json.loads(config)
        print(f"config_dict: {config_dict}")

        if "config" in config_dict:
            config_list = config_dict["config"]
            print(f"config['config']: {config_list}")
            print(f"Type of config['config']: {type(config_list)}")

            # If you need to remove an item from the list, use indexing
            batch_input_shape = None
            for item in config_list:
                if isinstance(item, dict) and "batch_input_shape" in item:
                    batch_input_shape = item.pop("batch_input_shape")
                    break

            print(f"batch_input_shape: {batch_input_shape}")
