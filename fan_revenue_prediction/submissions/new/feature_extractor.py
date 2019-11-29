import os
import pandas as pd
import numpy as np
import unidecode
import warnings
warnings.filterwarnings("ignore")


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):

        path = os.path.dirname(__file__)
        award = pd.read_csv(os.path.join(path, 'award_notices_RAMP.csv.zip'),
                            compression='zip', low_memory=False)
        # keeping first 2 numbers of APE
        
        X_df['Activity_code (APE)'] = X_df['Activity_code (APE)'].fillna(value = '00').astype('str').apply(lambda x: x[:2])
        
        # infer zipcode from city
        
        if(True):
            X_df['Zipcode'] = X_df['Zipcode'].astype('str').apply(lambda x: x[:2])
            city_names = X_df[['Zipcode', 'City']].groupby('Zipcode').City.apply(list).values
            zipcodes = X_df[['Zipcode', 'City']].groupby('Zipcode').City.apply(list).index
            zip_dict = {zipcode:list(set(cities)) for zipcode,cities in zip(zipcodes, city_names)}
            
            #remove nans from city lists
            
            for key in zip_dict.keys():
                for i, city in enumerate(zip_dict[key]):
                    if type(city) != type('a string'):
                        zip_dict[key].pop(i)
                   
            # infer zip by checking lists
            
            def infer_zip(city):
                for key in zip_dict:
                    if city in zip_dict[key]:
                        return key
            
            X_df.loc[X_df['Zipcode'] == 'na', 'Zipcode'] = X_df.loc[X_df['Zipcode'] == 'na', 'City'].apply(infer_zip)
            X_df.loc[X_df['Zipcode'] == 'na', 'Zipcode'] = '00'
            X_df['Zipcode'] = X_df['Zipcode'].fillna('00')
            
        # fill name and lower
        
        X_df['Name'] = X_df['Name'].fillna('nan').str.lower().astype('str').apply(unidecode.unidecode)
        
        # Fill headcounts
        
        if(True):
            headcounts = X_df[['Name', 'Headcount']].groupby('Name').mean().fillna(0).to_dict()['Headcount']
            names = X_df['Name'].values
            heads = np.zeros(len(names))
        
            for i,name in enumerate(names):
                if type(name) == type('a string'):
                    heads[i] = headcounts[name]
        
            X_df['Headcount'] = heads
        
        # unicode and lower
        
        X_df['City'] = X_df['City'].fillna('nan').str.lower().apply(unidecode.unidecode)
        
        city_list = [unidecode.unidecode(i.lower()) for i in list(set(X_df['City'].values))]
        
        def cedex_remover(s):
            s = s.replace('-', ' ').replace(',', ' ')
            tokens = s.split(' ')
            for k in range(len(tokens)):
                for j in range(1, len(tokens)-k + 1):
                    if ' '.join(tokens[k:k+j]) in city_list:
                        return ' '.join(tokens[k:k+j])
            return None
        
        
        # unidecode incumbent_name and lower
        
        award['incumbent_name'] = award['incumbent_name'].str.lower().astype('str').apply(unidecode.unidecode)
        
        # preprocess incumbent_city and fill nans
        award['incumbent_city'] = award['incumbent_city'].str.lower().astype('str').apply(unidecode.unidecode).apply(cedex_remover).fillna('nan')
        
        
        # Remove NA in department of provision and keep first 2
        award = award[award['Departments_of_publication'].notna()]
        award['Departments_of_publication'] = award['Departments_of_publication'].astype('str').apply(lambda x: x.split(',')[0][:2])
        
        # city mean FAN

        city_mean_fan = award[['incumbent_city','amount']].groupby('incumbent_city').quantile(0.5).fillna(0).to_dict()['amount']
        
        city_lower_mean = award[['incumbent_city','amount']].groupby('incumbent_city').quantile(0.25).fillna(0).to_dict()['amount']
        city_higher_mean = award[['incumbent_city','amount']].groupby('incumbent_city').quantile(0.75).fillna(0).to_dict()['amount']
        
        # zip mean FAN

        zip_mean_fan = award[['Departments_of_publication','amount']].groupby('Departments_of_publication').quantile(0.5).fillna(0).to_dict()['amount']
        
        zip_lower_mean = award[['Departments_of_publication','amount']].groupby('Departments_of_publication').quantile(0.25).fillna(0).to_dict()['amount']
        zip_higher_mean = award[['Departments_of_publication','amount']].groupby('Departments_of_publication').quantile(0.75).fillna(0).to_dict()['amount']
        
        # APE mean FAN
        company_list = [unidecode.unidecode(str(i).lower()) for i in list(set(X_df['Name'].values))]
        
        company_awards = award[award['incumbent_name'].isin(company_list)]
        
        ape_companies = X_df[['Name', 'Activity_code (APE)']].groupby('Name')['Activity_code (APE)'].apply(lambda x: list(x)[0])
        ape_companies.index = ape_companies.index.str.lower()
        ape_companies = ape_companies.to_dict()
        
        def infer_ape(name):
            return ape_companies[name]
        company_awards['APE'] = company_awards['incumbent_name'].apply(infer_ape)
        
        ape_mean_fan = company_awards[['APE','amount']].groupby('APE').quantile(0.5).fillna(0).to_dict()['amount']
        ape_lower_mean = company_awards[['APE','amount']].groupby('APE').quantile(0.25).fillna(0).to_dict()['amount']
        ape_higher_mean = company_awards[['APE','amount']].groupby('APE').quantile(0.75).fillna(0).to_dict()['amount']
        
        # Actual FAN revenue

        fan_revenue = company_awards[['incumbent_name','amount']].groupby('incumbent_name').sum().fillna(0).to_dict()['amount']

        # Insert features into X_df
        
        # Ape FAN
        if(True):
            def infer_ape_fan(APE):
                try:
                    return ape_mean_fan[APE]
                except:
                    return 0
        
            def infer_low_ape_fan(APE):
                try:
                    return ape_lower_mean[APE]
                except:
                    return 0
        
            def infer_high_ape_fan(APE):
                try:
                    return ape_higher_mean[APE]
                except:
                    return 0
        
        
        
            X_df['APE_fan_0.5'] = X_df['Activity_code (APE)'].apply(infer_ape_fan)
            X_df['APE_fan_0.25'] = X_df['Activity_code (APE)'].apply(infer_low_ape_fan)
            X_df['APE_fan_0.75'] = X_df['Activity_code (APE)'].apply(infer_high_ape_fan)
        
        # City FAN
        if(True):
        
            def infer_city_fan(city):
                try:
                    return city_mean_fan[city]
                except:
                    return 0
        
            def infer_low_city_fan(city):
                try:
                    return city_lower_mean[city]
                except:
                    return 0
        
            def infer_high_city_fan(city):
                try:
                    return city_higher_mean[city]
                except:
                    return 0
        
        
            X_df['city_fan_0.5'] = X_df['City'].apply(infer_city_fan)
            X_df['city_fan_0.25'] = X_df['City'].apply(infer_low_city_fan)
            X_df['city_fan_0.75'] = X_df['City'].apply(infer_high_city_fan)
        
        # Zip_fan
        if(True):
            def infer_zip_fan(zipcode):
                try:
                    return zip_mean_fan[zipcode]
                except:
                    return 0
        
            def infer_low_zip_fan(zipcode):
                try:
                    return zip_lower_mean[zipcode]
                except:
                    return 0
        
            def infer_high_zip_fan(zipcode):
                try:
                    return zip_higher_mean[zipcode]
                except:
                    return 0
        
            X_df['zip_fan_0.5'] = X_df['Zipcode'].apply(infer_zip_fan)
            X_df['zip_fan_0.25'] = X_df['Zipcode'].apply(infer_low_zip_fan)
            X_df['zip_fan_0.75'] = X_df['Zipcode'].apply(infer_high_zip_fan)
        
        # Actual FAN
        if(True):
            def infer_fan(name):
                try:
                    return fan_revenue[name]
                except:
                    return 0
        
            X_df['FAN'] = X_df['Name'].apply(infer_fan)
        
        # Returning X_array
        to_dummy = ['Activity_code (APE)', 'Zipcode', 'Year']
        to_drop = ['Legal_ID','Name', 'Address', 'City', 'Fiscal_year_end_date']
        X_array = pd.get_dummies(X_df, columns = to_dummy).drop(to_drop, axis = 1)
        return X_array