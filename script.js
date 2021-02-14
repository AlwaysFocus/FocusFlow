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

    // Convert data to a form that can be trained
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;

    // Train the model
    await trainModel(model, inputs, labels);
    console.log('Training finished!')
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


/**
 * Convert data to tensors that can be used for machine learning
 * Also going to perform shuffling and normalizing on the MPG data
 */

 function convertToTensor(data) {
     // The .tidy method will dispose of 
     // intermediate tensors

     return tf.tidy(() => {
         // Shuffle the data - this should always be done before 
         // passing the data to the training algorithms
         tf.util.shuffle(data);

         // Convert the data to Tensor
         const inputs = data.map(data => data.horsepower);
         const labels = data.map(data => data.mpg)

         const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
         const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

         // Normalize the data to fall within a range of 0-1 using min-max scaling
         const inputMax = inputTensor.max();
         const inputMin = inputTensor.min();
         const labelMax = labelTensor.max();
         const labelMin = labelTensor.min();

         const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
         const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

         return {
             inputs: normalizedInputs,
             labels: normalizedLabels,
             // Min max boundaries for later use
             inputMax,
             inputMin,
             labelMax,
             labelMin,
         }
         
     });
 }

 async function trainModel(model, inputs, labels) {
     // Prep model for training
     model.compile({
         optimizer: tf.train.adam(),
         loss: tf.losses.meanSquaredError,
         metrics: ['mse'],
     });

     const batchSize = 32;
     const epochs = 50;

     return await model.fit(inputs, labels, {
         batchSize,
         epochs,
         shuffle: true,
         callbacks: tfvis.show.fitCallbacks(
             {name: 'Model Training Performance'},
             ['loss', 'mse'],
             {height: 200, callbacks: ['onEpochEnd']}
             )
     });
 }


document.addEventListener('DOMContentLoaded', run);