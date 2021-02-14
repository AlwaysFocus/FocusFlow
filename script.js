/**
 * Retrieve car data and reduce it to just the data
 * values that have no null values
 */
async function  getData() {
    const carDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carDataResponse.json();
    const cleanedData = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
    .filter(car => (car.mpg != null && car.horsepower != null));

    return cleanedData;
}

async function run() {
    // Load and plot data we will perform training with
    const data = await getData();
    const values = data.map(data => ({
        y: data.mpg,
        x: data.horsepower,
    }));

    // Render the data onto a scatter plot map to see if it can be used 
    // to train our model
    tfvis.render.scatterplot(
        {name: 'HP vs MPG'},
        {values},
        {
            xLabel: 'Horsepower',
            yLabel: 'Miles per Gallon',
            height: 300
        }
    );

    // Create the model
    const model = createModel();
    tfvis.show.modelSummary({name: 'Model Summary'}, model);
}


/**
 * Define our model architecture
 */
function createModel() {
    // Create a sequential model which will have input flow directly down to output
    const model = tf.sequential();

    // Add a single input layer to our model
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

    // Add an output layer to our model
    /**
     * Because the hidden layer has 1 unit, we don't actually need to add 
     * the final output layer below (i.e. we could use hidden layer as the output layer). 
     * However, defining a separate output layer allows us to modify the number of units 
     * in the hidden layer while keeping the one-to-one mapping of input and output.
     */
    model.add(tf.layers.dense({units: 1, useBias: true}));

    return model;
}


document.addEventListener('DOMContentLoaded', run);