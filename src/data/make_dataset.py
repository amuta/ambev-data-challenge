# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np

@click.command()
@click.argument('data_step', type=click.Path())
def main(data_step):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    global logger
    logger = logging.getLogger(__name__)
    print(data_step)
    assert data_step in ['all', 'interim']
    logger.info('making {} data'.format(data_step))


    AMBEV_FILE = 'ambev-final-dataset.csv'
    RAW_PATH = 'data/raw/'
    INTE_PATH = 'data/interim/'
    PROC_PATH = 'data/processed/'

    # Create interim file if not exists
    if os.path.isfile(INTE_PATH + AMBEV_FILE):
        logger.info('using found interim file {}'.format(AMBEV_FILE))
    else:
        logger.info('creating interim file {}'.format(AMBEV_FILE))
        raw_df = pd.read_csv(RAW_PATH + AMBEV_FILE)
        int_df = make_interim(raw_df)
        int_df.to_csv(INTE_PATH + AMBEV_FILE)

    # Create processed file if not exists


def make_interim(df):
    """ pipeline of interim df"""
    df = fix_colnames(df)
    df = fix_cols(df)
    return df


def fix_colnames(df):
    # new column names
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
    """ Fix unaligned columns and column names
    """
    logger.info('Fixing columns.')

    # fixing columns
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




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
