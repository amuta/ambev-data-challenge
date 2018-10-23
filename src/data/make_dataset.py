# -*- coding: utf-8 -*-
# import click
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# @click.command()
# @click.argument('data_step', type=click.Path())


def main(force=False, out=False):
    ''' Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    '''
    global logger
    logger = logging.getLogger(__name__)
    logger.info('start making data...')

    AMBEV_FILE = 'ambev-final-dataset.csv'
    RAW_PATH = 'data/raw/'
    INTE_PATH = 'data/interim/'
    PROC_PATH = 'data/processed/'

    processed_file = PROC_PATH + AMBEV_FILE.replace('.csv', '.pkl')
    interim_file = INTE_PATH + AMBEV_FILE.replace('.csv', '.pkl')
    raw_file = RAW_PATH + AMBEV_FILE

    # Create interim file if not exists
    if os.path.isfile(interim_file) and not force:
        logger.info('using found interim file {}'.format(interim_file))
        df = pd.read_pickle(interim_file)
    else:
        logger.info('creating interim file {}'.format(interim_file))
        df = pd.read_csv(raw_file, encoding='utf-8', low_memory=False)
        df = make_interim(df)
        df.to_csv(interim_file, index=False)
        df.to_pickle(interim_file)

    # Create processed file if not exists
    if os.path.isfile(processed_file) and not force:
        logger.info('using found processed file {}'.format(processed_file))
        df = pd.read_pickle(processed_file)
    else:
        logger.info('creating processed file {}'.format(processed_file))
        df = pd.read_pickle(interim_file)
        df = make_processed(df)
        df.to_csv(processed_file, index=False)
        df.to_pickle(processed_file)

    # Finishing up
    logger.info('make data ran successfully\n interin file is located at {}'
                '\n processed file is located at {}'.format(interim_file,
                                                            processed_file))

    # Returning if called outside make
    if out:
        return df


######################
# INTERIM DATA PIPELINE
######################
def make_interim(df):
    ''' pipeline of interim df'''
    logger.info('fixing column names')
    df = fix_colnames(df)
    logger.info('fixing column alignment')
    df = fix_cols(df)
    logger.info('correcting dtypes')
    df = correct_dtypes(df)
    logger.info('cleaning values')
    df = clean_values(df)
    logger.info('creating new features')
    df = new_features(df)
    return df


def fix_colnames(df):
    ''' Fix problematic column names
    '''
    col_names = [
        'mes', 'pais', 'mundo', 'area_regional', 'unidade',
        'grupo_cargo', 'cargo', 'grade', 'banda', 'area', 'id_funcionario',
        'id_gestor', 'id_kpi', 'diretoria', 'area_diretoria',
        'funcao', 'tipo_meta', 'categoria_kpi', 'nome_kpi', 'peso_kpi',
        'prazo', 'regra_alcance_parcial', 'meta_projeto', 'ating_mes',
        'pts_mes', 'acum_mes', 'ating_acum', 'pts_acumulado',
        'acum_acumulado', 'ating_fim_exer', 'pts_fim_exer',
        'acum_fim_exer', 'status_meta', 'c33', 'c34', 'c35', 'c36', 'c37']

    df.columns = col_names
    df['ult_col'] = np.nan
    return df


def fix_cols(df):
    ''' Fix unaligned columns and column names
    '''
    bad_cols = df.columns.values[21:]
    for i in range(6):
        # meta_projeto should always be s/n/nan
        idxs = df[~df.meta_projeto.isin([
            'Sim', 'Não', np.nan, np.NAN])].index
        df.loc[idxs, 'regra_alcance_parcial'] = (
            df.loc[idxs, 'regra_alcance_parcial'].fillna('').astype(str) +
            ' ' + df.loc[idxs, 'meta_projeto'].fillna('').astype(str))
        for i in range(1, len(bad_cols) - 1):
            df.loc[idxs, bad_cols[i]] = df.loc[idxs, bad_cols[i + 1]].values

    # dropping now empty columns
    df.drop(['c33', 'c34', 'c35', 'c36', 'c37', 'ult_col'],
            axis=1, inplace=True)
    return df


def correct_dtypes(df):
    ''' Change the dtypes of columns to something more relatable
    '''
    df[df.columns[:23]] = df[df.columns[:23]].astype('str')
    df.status_meta = df['status_meta'].astype('str')
    df[df.columns[23:32]] = df[df.columns[23:32]].astype(np.number)
    return df


def clean_values(df):
    ''' Do some data cleaning on problematic columns
    '''
    # Format errors
    df.mes = df.mes.astype(str).str.extract('(^.*)(?=.{4}$)').astype(int)
    # Encoding errors
    df.loc[df.pais == 'PanamÁ¡', 'pais'] = 'Panama'
    # NLP transforms
    cols = [
        'cargo', 'grupo_cargo', 'area', 'diretoria', 'mundo',
        'area_diretoria', 'funcao', 'banda', 'tipo_meta', 'categoria_kpi',
        'regra_alcance_parcial']
    for col in cols:
        df[col] = df[col].str.lower().str.replace(r'[^\w\s]+', '')
        df[col] = df[col].str.normalize('NFKD').str.encode(
            'ascii', errors='ignore').str.decode('utf-8')

    # Month cleanup
    df.prazo = df.prazo.astype('str').apply(extract_month)
    return df


def extract_month(text):
    ''' Extract the month of some unformated date value using regex
    '''
    if re.match(r'(^[0-3]?[0-9].[0-1]?[0-9].?[0-9]{4})', text):
        val = re.search(r'^[0-3]?[0-9].([0-1]?[0-9])', text).groups()[0]
    elif re.match(r'(^[0-1]?[0-9].[0-9]?[0-9].?[0-9]{4})', text):
        val = re.search(r'^([0-1]?[0-9])', text).groups()[0]
    elif re.match(r'(^[0-9]?[0-9](?:[.]|[\/])[0-1]?[0-9].?[0-9]{2})', text):
        val = re.search(r'^[0-3]?[0-9].([0-1]?[0-9])', text).groups()[0]
    elif re.match(r'monthly', text, re.IGNORECASE):  # not really a date
        val = 0
    elif re.match(r'^([0-9]{5})', text):  # excel date format
        val = datetime.fromordinal(
            datetime(1900, 1, 1).toordinal() + int(
                re.search(r'^([0-9]{5})', text).groups()[0]) - 2).month
    else:
        val = 0
    return val


def new_features(df):
    ''' Create new features
    '''
    df['abrev_cargo'] = df.cargo.str[:3]
    df['abrev_grupo_cargo'] = df.grupo_cargo.str[:3]
    df['nivel_cargo'] = df.cargo.str.extract(r'([iv]{1,3}$)')
    df['regra_n1'] = df.regra_alcance_parcial.str.extract(r'(\d{2})')
    df['regra_n2'] = df.regra_alcance_parcial.str.extract(r'(?:\d{2})(?:.*)(\d{2})')
    df['regra_n3'] = df.regra_alcance_parcial.str.extract(r'(?:\d{2})(?:.*)(?:\d{2})(?:.*)(\d{2})')
    df['regra_n4'] = df.regra_alcance_parcial.str.extract(r'(?:\d{2})(?:.*)(?:\d{2})(?:.*)(?:\d{2})(?:.*)(\d{2})')
    df['regra_real'] = df.regra_alcance_parcial.str.contains('real')
    df['regra_lacuna'] = df.regra_alcance_parcial.str.contains('lacuna')
    df['regra_pontosl'] = df.regra_alcance_parcial.str.contains('pontos')
    return df


######################
# PROCESSED DATA PIPELINE
######################
def make_processed(df):
    ''' Final cleanup of processed data for model input
    '''
    logger.info('removing bad data')
    df = remove_bad_data(df)
    logger.info('making target')
    df = make_target(df)
    logger.info('removing ids and using first row')
    df = remove_ids_months(df)
    logger.info('encoding categorical columns')
    df = encode_categoricals(df)
    return df


def remove_bad_data(df):
    ''' Remove non usable columns and data
    '''
    # correlated/leaky columns
    col_names = ['ating_acum', 'ating_fim_exer', 'pts_mes', 'pts_acumulado',
                 'pts_fim_exer', 'acum_mes', 'acum_acumulado', 'acum_fim_exer']
    df = df[df.status_meta.notna()]
    df = df.fillna(0)
    df = df[df.columns[df.nunique() > 1]]
    df = df.drop(col_names, axis=1)
    return df


def make_target(df):
    ''' Make the target variable
    '''
    # df['ating_mes'] = df.ating_mes.astype(np.number)
    df['target'] = df.groupby(
        ['id_funcionario']).ating_mes.transform('mean') / 100
    df = df.drop([
        'mes', 'ating_mes', 'nome_kpi', 'regra_alcance_parcial'],
        axis=1)
    
    return df


def remove_ids_months(df):
    ''' Remove unique identifiers and use only first month data
    '''
    df = df.groupby(['id_funcionario']).agg('first').reset_index()
    df = df.drop(['id_funcionario', 'id_gestor'], axis=1)  # remove ids
    return df


def encode_categoricals(df):
    ''' Encode categorical data
    '''
    le = LabelEncoder()
    # columns to envcode (non-numeric)
    col_names = df.drop('target', axis=1).columns

    for col in col_names:
        # logger.info('column {}'.format(col))
        df[col] = le.fit_transform(df[col].astype('str'))
    return df


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(force=True)
