# dashapp: turn your pandas dataframes into websites

dashapp is a python library that allows you to assemble Plotly Dash widgets, whose source data comes from Pandas DataFrames.  

Each component that you create takes a DataFrame (or slice of a DataFrame) as an input argument, and returns a Dash Component.  These components can be assembled into panels.  The panels can be assembled into columns, rows or rows of columns.

Additionally, component can be linked together so that state changes in one component can generate reactions and state changes in linked components.