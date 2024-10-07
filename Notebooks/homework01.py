import marimo

__generated_with = "0.8.15"
app = marimo.App(width="medium", app_title="ml-zoomcamp-homework01")


@app.cell
def __(mo):
    mo.md(
        r"""
        This is to follow the ML Zoomcamp course.

        I will also use it to learn bells and whistles, such as polars and marimo notebooks.

        Good polars cheat sheets:

        - https://www.rhosignal.com/posts/polars-pandas-cheatsheet/
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    return cs, mo, np, pd, pl


@app.cell
def __(mo, pd):
    mo.md(f"""Which pandas version? 

    {pd.__version__}""")
    return


@app.cell
def __(mo, pl):
    mo.md(f"""Which polars version? 

    {pl.__version__}""")


    return


@app.cell
def __(pl):
    laptops = pl.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv")
    laptops.head(2)
    return laptops,


@app.cell
def __(laptops, mo):
    mo.md(f""" How many rows?

    {laptops.shape[0]}""")
    return


@app.cell
def __(laptops, mo):
    #laptops["Brand"].value_counts().shape #pandas way
    brands = laptops.select('Brand').unique().shape[0] #polars way
    mo.md(f"""How many brands of laptops?

    {brands}""")
    return brands,


@app.cell
def __(laptops, mo, pl):
    nul_col = laptops.null_count().transpose().filter(pl.col('column_0')>0).shape[0]

    mo.md(f"""How many columns have null?

    {nul_col}""")
    return nul_col,


@app.cell
def __(laptops, mo, pl):

    ans = laptops.filter(pl.col('Brand')=='Dell').select('Final Price').max()[0,0]

    mo.md(f"""What's the maximum final price of Dell notebooks in the dataset?

    {int(ans)}""")
    return ans,


@app.cell
def __():
    return


@app.cell
def __(laptops, mo):
    # mode and median on a series: use [] to convert to Series in pl
    # mode returns multiple values (because it can be bimodal etc.)
    med = laptops["Screen"].median()
    mod = laptops["Screen"].mode().to_list()

    mo.md(f"""Median and mode value of screen column before NA removal:

    median: {med}

    mode: {mod}""")
    return med, mod


@app.cell
def __(laptops, mo, mod):
    # mode and median on a series: use [] to convert to Series in pl
    # mode returns multiple values (because it can be bimodal etc.)

    newmed = laptops["Screen"].fill_nan(mod[0]).median()
    newmod = laptops["Screen"].fill_nan(mod[0]).mode().to_list()

    mo.md(f"""Median and mode value of screen column after NA replacement with mode:

    median: {newmed}

    mode: {newmod}""")
    return newmed, newmod


@app.cell
def __(laptops, np):
    # make a numpy array
    X = laptops.filter(Brand = 'Innjoo').select("RAM", "Storage", "Screen").to_numpy()

    # transpose
    X.T

    # multiply
    # The numpy.matmul function implements the @ operator.
    XTX = X.T @ X

    # inverse
    Xi = np.linalg.inv(XTX)

    # new array
    y = np.array([1100, 1300, 800, 900, 1000, 1100])

    #Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w
    w = (Xi @ X.T) @ y

    #What's the sum of all the elements of the result?

    return X, XTX, Xi, w, y


@app.cell
def __(mo, w):
    mo.md(f"""Implementing linear regressison with matrix multiplication:

    {sum(w)}""")
    return


if __name__ == "__main__":
    app.run()
