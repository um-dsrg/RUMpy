const fs = require('fs');
const { parse } = require('csv-parse');
const path = require('path');
const csvFilename = path.resolve(__dirname, '..', 'models.csv');

const models = document.getElementById('models');
const modelStatus = document.getElementById('modelStatus');
const loadModel = document.getElementById('loadModel');
const processor = document.getElementById('processor');
const metadata = document.getElementById('metadata');
const apiUrl = document.getElementById('url');
const videoFps = document.getElementById('videoFps');
const connection = document.getElementById('testConnection');
const connectionStatus = document.getElementById('connectionStatus');

var loadedModelsList = [];

function setInitialValues() {
    if (!sessionStorage.getItem('model')) {
        sessionStorage.setItem('model', models.value);
        sessionStorage.setItem('modelName', models[models.value].text);
    }

    if (!sessionStorage.getItem('processor')) {
        sessionStorage.setItem('processor', processor.value);
    }

    if (!sessionStorage.getItem('metadata')) {
        sessionStorage.setItem('metadata', metadata.value);
    }

    if (!sessionStorage.getItem('url')) {
        sessionStorage.setItem('url', apiUrl.value);
    }

    if (!sessionStorage.getItem('videoFps')) {
        sessionStorage.setItem('videoFps', videoFps.value);
    }

    if (!sessionStorage.getItem('connection')) {
        sessionStorage.setItem('connection', connectionStatus.innerText);
    }

    if (!sessionStorage.getItem('connectionColour')) {
        sessionStorage.setItem('connectionColour', connectionStatus.style.color);
    }

    if (!sessionStorage.getItem('modelText')) {
        sessionStorage.setItem('modelText', modelStatus.innerText);
    }

    if (!sessionStorage.getItem('modelColour')) {
        sessionStorage.setItem('modelColour', modelStatus.style.color);
    }
}

function saveValues(event) {
    sessionStorage.setItem(event.target.id, event.target.value);
}

function getSavedValues() {
    if (sessionStorage.getItem('model')) {
        models.value = sessionStorage.getItem('model');
    }

    if (sessionStorage.getItem('processor')) {
        processor.value = sessionStorage.getItem('processor');
    }

    if (sessionStorage.getItem('metadata')) {
        metadata.value = sessionStorage.getItem('metadata');
    }

    if (sessionStorage.getItem('url')) {
        apiUrl.value = sessionStorage.getItem('url');
    }

    if (sessionStorage.getItem('videoFps')) {
        videoFps.value = sessionStorage.getItem('videoFps');
    }

    if (sessionStorage.getItem('connection')) {
        connectionStatus.innerText = sessionStorage.getItem('connection');
    }

    if (sessionStorage.getItem('connectionColour')) {
        connectionStatus.style.color = sessionStorage.getItem('connectionColour');
    }

    if (sessionStorage.getItem('modelText')) {
        modelStatus.innerText = sessionStorage.getItem('modelText');
    }

    if (sessionStorage.getItem('modelColour')) {
        modelStatus.style.color = sessionStorage.getItem('modelColour');
    }
}

function zeros(value) {
    if (value < 10) {
        return '0' + value;
    } else {
        return value;
    }
}

function testApiConnection() {
    $.get(apiUrl.value + 'test_page')
    .done(function(data) {
        var today = new Date();

        connectionStatus.innerText = 'Ok (Last tested: ' + zeros(today.getHours()) + ':' + zeros(today.getMinutes()) + ':' + zeros(today.getSeconds()) + ')';
        connectionStatus.style.color = '#1ABD35';
        sessionStorage.setItem('connection', connectionStatus.innerText);
        sessionStorage.setItem('connectionColour', connectionStatus.style.color);
    })
    .fail(function(error) {
        var today = new Date();

        connectionStatus.innerText = 'Error (Last tested: ' + zeros(today.getHours()) + ':' + zeros(today.getMinutes()) + ':' + zeros(today.getSeconds()) + ')';
        connectionStatus.style.color = '#BD281A';
        sessionStorage.setItem('connection', connectionStatus.innerText);
        sessionStorage.setItem('connectionColour', connectionStatus.style.color);
    });
}

function updateModel(event) {
    var modelAndProcessor = Object.assign({}, loadedModelsList[models.value], {'processor': processor.value});
    $.post(apiUrl.value + 'update_model', modelAndProcessor)
    .done(function(data) {
        var today = new Date();

        modelStatus.innerText = 'Ok (Model loaded: ' + zeros(today.getHours()) + ':' + zeros(today.getMinutes()) + ':' + zeros(today.getSeconds()) + ')';
        modelStatus.style.color = '#1ABD35';
        sessionStorage.setItem('modelText', modelStatus.innerText);
        sessionStorage.setItem('modelColour', modelStatus.style.color);
        sessionStorage.setItem('processor', processor.value);
        sessionStorage.setItem('model', models.value);
        sessionStorage.setItem('modelName', models[models.value].text);
    })
    .fail(function(error) {
        var today = new Date();

        modelStatus.innerText = 'Error (Model not loaded: ' + zeros(today.getHours()) + ':' + zeros(today.getMinutes()) + ':' + zeros(today.getSeconds()) + ')';
        modelStatus.style.color = '#BD281A';
        sessionStorage.setItem('modelText', modelStatus.innerText);
        sessionStorage.setItem('modelColour', modelStatus.style.color);
    });
}

function addNewOption(event) {
    var id = 0;

    fs.createReadStream(csvFilename)
        .pipe(parse({ delimiter: ',', columns: true }))
        .on('data', function (row) {
            loadedModelsList.push(row);
        })
        .on('end', function(end) {
            var grouped = _.groupBy(loadedModelsList, model => model.group);
            _.forEach(grouped, function(value, key) {
                var newOptionGroup = document.createElement('optgroup');
                newOptionGroup.label = key;
                newOptionGroup.setAttribute('class', 'descItem');

                _.forEach(value, function(model) {
                    var newOption = document.createElement('option');
                    newOption.text = model['label'];
                    newOption.setAttribute('value', id);
                    newOptionGroup.appendChild(newOption);
                    id += 1;
                });
                models.add(newOptionGroup);
            });
            setInitialValues();
        })
        .on('error', function (error) {
            console.log(error.message);
        });
}

metadata.addEventListener('change', saveValues);

apiUrl.addEventListener('change', saveValues);

videoFps.addEventListener('change', saveValues);

connection.addEventListener('click', testApiConnection);

loadModel.addEventListener('click', updateModel);

window.addEventListener('pageshow', setInitialValues);

window.addEventListener('pageshow', addNewOption);

window.addEventListener('DOMContentLoaded', getSavedValues);
