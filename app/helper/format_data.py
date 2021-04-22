import pandas as pd

class FormatData:
    def merge_data():

        indo_hs = pd.read_csv("app/temp/indo_hs_sample.txt",
                              sep="\t", error_bad_lines=False)

        indo_hs.columns = ['label', 'text']

        indo_hs.label = indo_hs.label.replace(['Non_HS', 'HS'], [0, 1])
        indo_hs["lang"]= ['id' for _ in range(len(indo_hs))]

        fox_news = pd.read_csv("app/temp/fox_news_hs_sample.txt",
                               sep=":", names=['label', 'text'])
        fox_news["lang"]  = ['en' for _ in range(len(fox_news))]

        frames = [indo_hs, fox_news]
        mix_data = pd.concat(frames)
        mix_data = mix_data.reset_index(drop=True)

        print(mix_data)
        mix_data.to_csv('app/temp/mix_hs_sample.csv', index=False)
