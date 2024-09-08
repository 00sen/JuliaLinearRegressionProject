using Pkg

# Descargar e instalar paquetes (si no están ya instalados)
Pkg.add(["CSV", "DataFrames", "Statistics", "Dates",
    "Random", "Plots", "StatsPlots", "StatsBase"])

using CSV
using DataFrames
using Statistics
using Dates
using Random
using Plots
using StatsPlots
using StatsBase

# Preparar paths
path = "dataSets/damCombustible.csv"
cleaned_path = "dataSets/damCombustible_cleaned.csv"

# Corregir cada linea del CSV
function fix_csv_line(line::String)
    parts = split(line, ",")
    horometro_index = 3

    if length(parts) > horometro_index + 1 && !isempty(parts[horometro_index+1])
        while length(parts) > horometro_index + 2 && parts[horometro_index+2] == ""
            parts = vcat(parts[1:horometro_index+1], parts[horometro_index+3:end])
        end
    end

    return join(parts, ",")
end


function correct_csv_file(input_path::String, output_path::String)
    csv_lines = readlines(input_path)

    fixed_lines = [fix_csv_line(line) for line in csv_lines]

    open(output_path, "w") do file
        for line in fixed_lines
            println(file, line)
        end
    end
end

correct_csv_file(path, cleaned_path)

df = CSV.read(cleaned_path, DataFrame)

# Rename columns
rename!(df, Symbol("Nro.") => :Numero,
    :Vehículo => :Vehiculo, :Odómetro => :Odometro,
    :Horómetro => :Horometro,
    Symbol("Tanqueo Full") => :Tanque_Lleno,
    Symbol("Costo por Volumen") => :Costo_Por_Volumen,
    Symbol("Cant.") => :Cantidad,
    Symbol("Costo Total") => :Costo_Total)

# Drop useless Columns
select!(df, Not(:Column12, :Horometro, :Unidad, :Tipo))

# Convert S and N to 1 and 0 respectively
# df.Tanque_Lleno = df.Tanque_Lleno .== "S"
transform!(df, :Tanque_Lleno => ByRow(x -> x == "S" ? 1 : 0) => :Tanque_Lleno)

#---------------------------------------------------------------

# Date convertion 
# Change spanish name to english
function replaceMonths(date)
    months = Dict(
        "ene." => "01", "feb." => "02", "mar." => "03", "abr." => "04",
        "may." => "05", "jun." => "06", "jul." => "07", "ago." => "08",
        "sep." => "09", "oct." => "10", "nov." => "11", "dic." => "12"
    )

    for (mes, month) in months
        if occursin(mes, date)
            return replace(date, mes => month)
        end
    end

    return date
end

df.Fecha = replaceMonths.(df.Fecha)

df.Fecha = Dates.DateTime.(df.Fecha, "dd/mm/yyyy HH:MM:SS")

#---------------------------------------------------------------

# Columns to convert with commas
columns_with_commas_to_convert = [:Costo_Por_Volumen, :Cantidad, :Costo_Total]

# Replace columns with dots and to float
for col in columns_with_commas_to_convert
    df[!, col] = replace.(df[!, col], "," => ".")
    df[!, col] = parse.(Float32, df[!, col])
end

# Firts we drop Fecha and Numero columns
df = select(df, Not(:Numero, :Fecha))

# Mezclar los índices aleatoriamente
shuffled_df = df[shuffle(1:nrow(df)), :]

# Function to normalize data
function normalize(column)
    min_val = minimum(column)
    max_val = maximum(column)
    return (column .- min_val) ./ (max_val .- min_val)
end

# Function to standardize data (Non-used, no normal distribution. No mean = 0 and std = 1)
function standardize(column)
    mean_val = mean(column)
    std_val = std(column)
    return (column .- mean_val) ./ std_val
end

# Funciton to normalize VECTORS
function normalize_vector(vector)
    min_val = minimum(vector)
    max_val = maximum(vector)
    return (vector .- min_val) ./ (max_val .- min_val)
end

transform!(shuffled_df, :Odometro => normalize => :Odometro_Norm,
    :Costo_Por_Volumen => normalize => :Costo_Por_Volumen_Norm,
    :Cantidad => normalize => :Cantidad_Norm,
    :Costo_Total => normalize => :Costo_Total_Norm)

# creamod un hyp_df con las columnas a usar
# Vehiculo
# Odometro_Norm
# Cantidad_Norm
hyp_df = select(shuffled_df, [:Odometro_Norm, :Cantidad_Norm])

# Creo un df aplicando One Hot Encoding para poder separar los vehiculos y entrenar varios modelos
trucks = 101:110

y = DataFrame([Symbol("Vehiculo_$(truck)") => (shuffled_df.Vehiculo .== truck) .|> Int for truck in trucks])

# Declaramos la funcion de hipotesis que va a correr todo para la unica neurona
function hyp(bias::Float64, hyp_df::DataFrame, i::Int64)
    global w
    y = 0.0
    # y = w1*x1 + w2*x2
    y = hyp_df.Odometro_Norm[i] * w[1] + hyp_df.Cantidad_Norm[i] * w[2]

    # Sumamos el bias
    y += bias

    # Funcion sigmoide
    y = 1 / (1 + ℯ^(-y))

    # retornamos el resultado de la fila
    return y

end

# Funcion para imprimir
function printInfo(epochs::Int64)

    global errors
    global w

    println("Epoch: ", epochs)
    println("Theta: ", w)
    println("Error: ", errors[epochs])
end

# Funcion para calcular el ajuste de cada w
function adjustNumberCalculation(y::Vector{Int64}, row::Int64, hyp::Float64, hyp_df::DataFrame, wIndex::Int64)

    w_adjustment = -1 * (y[row] * hyp) * hyp * (1 - hyp) * hyp_df[row, wIndex]

end

# Declaramos el vector para guardar el historial de errores
global errors = Vector{Float64}()

function GD(epochs::Int64, alpha::Float64, hyp_df::DataFrame, y::Vector{Int64})
    global w
    global bias
    global errors

    temp_error_of_epoch = 0.0

    for row in 1:nrow(hyp_df)

        # Sacamos la hipostesis de la row
        hyp_result = hyp(bias, hyp_df, row)

        # Sacamos el error de la hyp
        hyp_error = hyp_result - y[row]
        temp_error_of_epoch += hyp_error

        # Calculamos error de la row y ajustamos w
        for w_index in 1:length(w)
            # Sacamos el error de la row con respecto a w
            adjust_number = adjustNumberCalculation(y, row, hyp_result, hyp_df, w_index)
            w[w_index] = w[w_index] - alpha * adjust_number
        end

    end



    # Sacamos el error promedio del epoch y lo guardamos en el historial de errores
    epoch_mean_error = temp_error_of_epoch / nrow(hyp_df)
    push!(errors, epoch_mean_error)

end

global model = 1

# 90% for training
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

@assert train_ratio + validation_ratio + test_ratio == 1.0

# Calcular el número de muestras para el conjunto de entrenamiento
n_train = floor(Int, train_ratio * nrow(hyp_df))
n_val = floor(Int, validation_ratio * nrow(hyp_df))

# Separar los índices en entrenamiento y prueba
train_hyp_df = hyp_df[1:n_train, :]
validation_hyp_df = hyp_df[(n_train+1):(n_train+n_val), :]
test_hyp_df = hyp_df[(n_train+n_val+1):end, :]

train_y = y[1:n_train, model]

validation_y = y[(n_train+1):(n_train+n_val), model]

test_y = y[(n_train+n_val+1):end, model]

epochs = 3000
alpha = 0.0001
global w = [float(rand(-1:1)), float(rand(-1:1))]
global bias = 0.0

for epoch in 1:epochs
    GD(epoch, alpha, train_hyp_df, train_y)
    printInfo(epoch)
end

global models = []
println(w)
push!(models, w)
models

# Declaramos la funcion de hipotesis que va a correr todo para la unica neurona
function hypTest(bias::Float64, validation_hyp_df::DataFrame, row::Int64)
    global model
    global models

    y = 0.0
    # y = w1*x1 + w2*x2
    y = validation_hyp_df.Odometro_Norm[row] * models[model][1] + validation_hyp_df.Cantidad_Norm[row] * models[model][2]

    # Sumamos el bias
    y += bias

    # Funcion sigmoide
    y = 1 / (1 + ℯ^(-y))

    # retornamos el resultado de la fila
    return y

end

validationResults = Vector{Int64}()

for row in 1:nrow(validation_hyp_df)
    if hypTest(bias, validation_hyp_df, row) > 0.5
        if validation_y[row] == 1
            push!(validationResults, 0)
        else
            push!(validationResults, 1)
        end
    elseif hypTest(bias, validation_hyp_df, row) < 0.5
        if validation_y[row] == 0
            push!(validationResults, 0)
        else
            push!(validationResults, 1)
        end
    end
end

count_ones = count(x -> x == 1, validationResults)
count_zeros = count(x -> x == 0, validationResults)

println("Datos validados correctamente: ", count_ones)
println("Datos validados INcorrectamente: ", count_zeros)

porcentaje_exito_validacion = ((count_ones * 100) / (count_ones + count_zeros))
println("Porcentaje de exito en la validacion: % ", porcentaje_exito_validacion)

testResults = Vector{Int64}()

for row in 1:nrow(test_hyp_df)
    if hypTest(bias, test_hyp_df, row) > 0.5
        if test_y[row] == 1
            push!(testResults, 0)
        else
            push!(testResults, 1)
        end
    elseif hypTest(bias, test_hyp_df, row) < 0.5
        if test_y[row] == 0
            push!(testResults, 0)
        else
            push!(testResults, 1)
        end
    end
end

count_ones = count(x -> x == 1, testResults)
count_zeros = count(x -> x == 0, testResults)

println("Datos predichos correctamente: ", count_ones)
println("Datos predichos INcorrectamente: ", count_zeros)

porcentaje_exito_test = ((count_ones * 100) / (count_ones + count_zeros))
println("Porcentaje de exito en el test: % ", porcentaje_exito_test)

heatMapGraphic = [[0, 0], [0, 0]]
for i in 1:length(testResults)
    if testResults[i] == 0
        if test_y[i] == 0
            heatMapGraphic[1][1] += 1
        else
            heatMapGraphic[1][2] += 1
        end
    else
        if test_y[i] == 0
            heatMapGraphic[2][1] += 1
        else
            heatMapGraphic[2][2] += 1
        end
    end
end

println("Matriz de confusion")
println(heatMapGraphic[1])
println(heatMapGraphic[2])

