import requests
import pandas as pd
import numpy as np

# alpha vantage api symbol search, returns best match symbol
def get_symbol():
    apikey = "XYCY71WIXA92A8FG"
    keyword = input("What ticker are you looking for?")
    print(f'Collecting Historical Data for {keyword}\n')
    url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={keyword}&apikey={apikey}'
    r = requests.get(url)
    data = r.json()
    print(data['bestMatches'][0])
    print('========================================================================================')
    return data['bestMatches'][0]['1. symbol']

# alpha vantage symbol search, returns list of match information
def get_symbol_info():
    apikey = "XYCY71WIXA92A8FG"
    keyword = input("What ticker are you looking for?")
    print(keyword)
    url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={keyword}&apikey={apikey}'
    r = requests.get(url)
    data = r.json()

    return data['bestMatches']

# alpha vantage api call, collects stock data based on symbol search, limits data to post 2009
def get_prices():
    apikey = "XYCY71WIXA92A8FG"
    symbol = get_symbol()

    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={apikey}&outputsize=full'
    r = requests.get(url)
    data = r.json()


    df = pd.DataFrame(data['Time Series (Daily)'])
    df = df.transpose()
    df.columns = columns=['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume', 'Dividend', 'Split']
    df = df.astype({'Open':'float', 'High':'float', 'Low':'float', 
                    'Close':'float', 'AdjClose':'float', 'Volume':'int', 'Dividend':'float', 'Split':'float'})
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df[df.index >= '2010']
    print(
        f'======================================================\n'
        f'Data Shape: {df.shape}\n'
        f'Date Range: {df.index.min()}  /  {df.index.max()}\n'
        f'All Time High: {df.High.max()}\n'
        f'All Time Low: {df.Low.min()}\n'
        f'======================================================\n'
        )


    return df

# Data Generator, creates and unrolls batches

class DataGeneratorSeq(object):

    def __init__(self,prices,batch_size,num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length //self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):

        batch_data = np.zeros((self._batch_size),dtype=np.float32)
        batch_labels = np.zeros((self._batch_size),dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b]+1>=self._prices_length:
                #self._cursor[b] = b * self._segments
                self._cursor[b] = np.random.randint(0,(b+1)*self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b]= self._prices[self._cursor[b]+np.random.randint(0,5)]

            self._cursor[b] = (self._cursor[b]+1)%self._prices_length

        return batch_data,batch_labels

    def unroll_batches(self):

        unroll_data,unroll_labels = [],[]
        init_data, init_label = None,None
        for ui in range(self._num_unroll):

            data, labels = self.next_batch()    

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0,min((b+1)*self._segments,self._prices_length-1))