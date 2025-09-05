module DataManager
using  CSV
using DataFrames

"""
Loads data from a CSV file into a DataFrame

Parameters:
- filename::AbstractString: path to the CSV file

Returns:
- DataFrame: loaded data
"""
function LoadData(filename::AbstractString)
    rawFile = CSV.File(filename)
    return DataFrame(rawFile)
end

"""
Adds delta columns (price changes) to the DataFrame

Parameters:
- df::DataFrame: input DataFrame with 'open' and 'close' columns

Returns:
- DataFrame: modified DataFrame with added 'deltaO' and 'deltaC' and 'deltaLogC' and 'deltaLogO' columns
the 'log' columns are subtractions of logarithms od respectable Open or Close columns
we also do the cleaning up of the missing values
"""
function AddDeltas!(df::DataFrame)
    select!(df, Not([:conversionSymbol, :conversionType]))
    df.deltaO = vcat([df.open[i+1] - df.open[i] for i in 1:(nrow(df)-1)], missing)
    df.deltaC = vcat([df.close[i+1] - df.close[i] for i in 1:(nrow(df)-1)], missing)
   # df.deltaLogO=vcat([log(df.open[i+1]) - log(df.open[i]) for i in 1:(nrow(df)-1)], missing)
   # df.deltaLogC=vcat([log(df.close[i+1]) - log(df.close[i]) for i in 1:(nrow(df)-1)], missing)
    dropmissing!(df,disallowmissing=true)

    df.deltaCJittered=df.deltaC.+1e-10.*randn(length(df.deltaC))
    df.deltaOJittered=df.deltaO.+1e-10.*randn(length(df.deltaO))
    
end




end