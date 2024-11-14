from flask import Flask, render_template, request, redirect, url_for
import ee
import datetime

# Initialize the Earth Engine API with your project ID
ee.Initialize(project='ee-shailendergoyal06')

app = Flask(__name__)

# Define the Bay of Bengal region
bay_of_bengal = ee.Geometry.Rectangle([76, 3, 100, 25])

# Visualization parameters (same as before)
vis_params = {
    'PS': {'min': 75000, 'max': 101500, 'palette': ['darkblue', 'blue', 'deepskyblue', 'cyan', 'lightcyan',
                                                     'green', 'yellow', 'orange', 'orangered', 'red', 'darkred']},
    'Q': {'min': 0, 'max': 0.1, 'palette': ['darkblue', 'blue', 'deepskyblue', 'cyan', 'lightcyan',
                                             'green', 'yellow', 'orange', 'orangered', 'red', 'darkred']},
    'RH': {'min': 0, 'max': 1.1, 'palette': ['darkblue', 'blue', 'deepskyblue', 'cyan', 'lightcyan',
                                              'green', 'yellow', 'orange', 'orangered', 'red', 'darkred']},
    'T': {'min': 250, 'max': 350, 'palette': ['darkblue', 'blue', 'deepskyblue', 'cyan', 'lightcyan',
                                               'green', 'yellow', 'orange', 'orangered', 'red', 'darkred']},
    'TPREC': {'min': 0, 'max': 0.05, 'palette': ['darkblue', 'blue', 'deepskyblue', 'cyan', 'lightcyan',
                                                  'green', 'yellow', 'orange', 'orangered', 'red', 'darkred']},
    'TROPPB': {'min': 6000, 'max': 39000, 'palette': ['darkblue', 'blue', 'deepskyblue', 'cyan', 'lightcyan',
                                                       'green', 'yellow', 'orange', 'orangered', 'red', 'darkred']},
    'TS': {'min': 260, 'max': 350, 'palette': ['darkblue', 'blue', 'deepskyblue', 'cyan', 'lightcyan',
                                                'green', 'yellow', 'orange', 'orangered', 'red', 'darkred']},
    'U': {'min': -50, 'max': 50, 'palette': ['purple', 'darkblue', 'blue', 'deepskyblue', 'cyan', 'lightcyan',
                                              'green', 'yellow', 'orange', 'orangered', 'red', 'darkred']},
    'V': {'min': -50, 'max': 50, 'palette': ['purple', 'darkblue', 'blue', 'deepskyblue', 'cyan', 'lightcyan',
                                              'green', 'yellow', 'orange', 'orangered', 'red', 'darkred']}
}

# List of variables to visualize
variables = ['PS', 'Q', 'RH', 'T', 'TPREC', 'TROPPB', 'TS', 'U', 'V']

# Define date ranges to process
date_ranges = [
    ('2018-01-01', '2024-11-11')
]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the selected date and hour from the form
        selected_date = request.form.get('date')
        selected_hour = request.form.get('hour')

        # Redirect to the images page with the selected date and hour
        return redirect(url_for('show_images', date=selected_date, hour=selected_hour))
    else:
        # Generate the list of dates and hours
        date_list = []
        for start_date_str, end_date_str in date_ranges:
            start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
            num_days = (end_date - start_date).days + 1
            for day_offset in range(num_days):
                date = start_date + datetime.timedelta(days=day_offset)
                date_str = date.strftime('%Y-%m-%d')
                date_list.append(date_str)
        date_list = sorted(list(set(date_list)))  # Remove duplicates and sort

        hours_list = [str(h).zfill(2) for h in range(24)]

        return render_template('index.html', date_list=date_list, hours_list=hours_list)


@app.route('/images')
def show_images():
    selected_date = request.args.get('date')
    selected_hour = request.args.get('hour')

    if not selected_date or not selected_hour:
        return redirect(url_for('index'))

    # Convert date and hour to datetime
    date_time_str = f"{selected_date} {selected_hour}:00:00"
    date_time = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')

    time_start = ee.Date(date_time)
    time_end = time_start.advance(1, 'hour')

    # Load the GEOS-CF dataset for the specific hour
    dataset = ee.ImageCollection("NASA/GEOS-CF/v1/rpl/tavg1hr") \
        .filterDate(time_start, time_end) \
        .filterBounds(bay_of_bengal)

    image = dataset.first()  # Get the first (and only) image for the hour

    if not image:
        return f"No data available for {selected_date} at {selected_hour}:00"

    # For each variable, get the image URL
    image_urls = {}
    for variable in variables:
        vis_param = vis_params[variable]
        image_rgb = image.select(variable).clip(bay_of_bengal).visualize(**vis_param)
        try:
            url = image_rgb.getThumbURL({
                'dimensions': [512, 512],  # Adjust dimensions as needed
                'region': bay_of_bengal.getInfo(),
                'format': 'png',
                'crs': 'EPSG:4326'
            })
            image_urls[variable] = url
        except Exception as e:
            image_urls[variable] = None

    # Render a template to display the images
    return render_template('images.html', date=selected_date, hour=selected_hour, image_urls=image_urls)


if __name__ == '__main__':
    app.run(debug=True)
