# A-Scan2BIM

Official implementation of the paper [A-Scan2BIM: Assistive Scan to Building Information Modeling](https://drive.google.com/file/d/1zvGfdlLYbd_oAp7Oc-1vF2Czhl1A7q75/view) (__BMVC 2023, oral__)

Please also visit our [project website](https://a-scan2bim.github.io/) for a video demo of our assistive system.

# Table of contents (also TODOs)

- [x] [Prerequisites](#prerequisites)
- [x] [Quickstart](#quickstart)
- [ ] [Training and evaluation](#training-and-evaluation)
- [ ] [Collecting your own data](#collecting-your-own-data)
- [ ] [Building and developing the plugin](#building-and-developing-the-plugin)
- [x] [Contact](#contact)
- [x] [Bibtex](#bibtex)
- [x] [Acknowledgment](#acknowledgment)

# Prerequisites

For more flexibility, our system is designed as a client-server model: a Revit plugin written in C# serves as the front end, and a python server which performs all the neural network computation.

To run our system, there are two options:

1. Run both the plugin and the server on the same machine
2. Run the server on a separate machine, and port-forward to client via local network or ssh

The advantage of option 2 over 1 is that the server can run on either Linux or Window machines, and can potentially serve multiple clients at the same time.
Of course, the client machine needs to be Windows as Revit is Windows-only.

The code has been tested with Revit 2022.
Other versions will likely work but are not verified.
If you are a student or an educator, you can get it for free [here](https://www.autodesk.com/education/edu-software/overview?sorting=featured&filters=individual).

# Quickstart

Follow this section to quickly install and run the assistive plugin on a sample point cloud.

Skip down to the section below for network training and evaluation details.

## Plugin installation (Windows machine)

Download the quickstart package [here](https://www.dropbox.com/scl/fi/jljkehuddx3df6hf6ptau/quickstart.zip?rlkey=bzxi1b13r00s6u29drkziazgv&dl=0), and unzip to somewhere on your computer. Yours should look like this:
```
.
├── ckpts/
├── code/
├── data/
├── plugins/
├── plugins_built/
└── ...
```

Install the Revit plugin by copying the content of `plugins_built/` to:
```
%APPDATA%\Autodesk\Revit\Addins\2022
```
You can navigate to that folder by copying the path above and paste into File Explorer's address bar. Change the year number at the end if you have a different Revit version.

Start up Revit.
You should see the warning below when you start up Revit for the first time after installing the plugin.
Click "Always Load" if you don't want to see this warning again.
If you do not see the warning, then the plugin has not been installed correctly.

![Addin warning](resources/addin_warning.png)

Open up `data/revit_projects/32_ShortOffice_05_F2.rvt` inside Revit.

Now you need to link the provided ReCap project, which contains the point cloud.

From the top ribbon menu, click on `Insert -> Manage Links -> Point Clouds -> 32_ShortOffice... -> Reload From...`.

Open `data/recap_projects/32_ShortOffice_05_F2/32_ShortOffice_05_F2_s0p01m.rcp`.

Finally, click "OK" and you should see the point cloud.

You are now ready from the plugin side.

## Server setup (Windows or Linux)

We use Miniconda 3 to manage the python environment, installed at `$HOME` directory.
Installation instructions can be found [here](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).

To install all dependencies, run the provided script: `sh ./setup_env.sh`.

Once the script finishes, run the following commands to start the server:
```
cd code/learn
conda activate bim
python backend.py
```

You are now ready from the server side.

## Using the assistive system

To make the plugin easier to use, you should setup some keyboard shortcuts.

From the top-left, click on `File -> Options -> User Interface -> Keyboard Shortcuts: Customize -> Filter: Add-Ins Tab`.
Bind the following commands to the corresponding keys:

| Command         | Shortcuts |
|----------------:|:---|
|Obtain Prediction| F2 |
|Add Corner       | F3 |
|Send Corners     | F4 |
|Next / Accept Green     | 4 |
|Reject / Accept Yellow    | 5 |
|Accept Red       | 6 |

To enable assistance, first go to the `Add-Ins` tab and click on `Autocomplete`.
The icon should change into a pause logo.

Now manually draw one wall, hit the Escape key twice to exit the "Modify Wall" mode, and you should see the next three suggested walls in solid (next) and dashed (subsequent) pink lines.

Run `Next / Accept Green` command to accept the solid pink line as the next wall to add.
You may interleave manual drawing or accept command however you like.

You may also run `Reject / Accept Yellow` to choose one of three candidate walls to add next.
Run the corresponding command to accept the colored suggestion.

To simplify wall drawing, one may also provide wall junctions and query the backend to automatically infer relevant walls.
This has the benefit of adding multiple walls at once, especially around higher-degree junctions.

To do so:
1. Hover the mouse over the junction in the point cloud, and hit the `Add Corner` shortcut.
2. (Optional) Drag the ring to modify its location.
3. Once the desired junctions are added, hit `Send Corners`.

# Training and evaluation

Coming soon...

# Collecting your own data

Coming soon...

# Building and developing the plugin

Coming soon...
# Contact

Weilian Song, weilians@sfu.ca

# Bibtex
```
@article{weilian2023ascan2bim,
  author    = {Song, Weilian and Luo, Jieliang and Zhao, Dale and Fu, Yan and Cheng, Chin-Yi and Furukawa, Yasutaka},
  title     = {A-Scan2BIM: Assistive Scan to Building Information Modeling},
  journal   = {British Machine Vision Conference (BMVC)},
  year      = {2023},
}
```

# Acknowledgment

This research is partially supported by NSERC Discovery Grants with Accelerator Supplements and the DND/NSERC Discovery Grant Supplement, NSERC Alliance Grants, and the John R. Evans Leaders Fund (JELF).

We are also grateful to the [CV4AEC CVPR workshop](https://cv4aec.github.io/) for providing the point clouds.

And finally, much of the plugin code was borrowed from Jeremy Tammik's sample add-in [WinTooltip](https://github.com/jeremytammik/WinTooltip).
Hats off :tophat: to Jeremy for all the wonderful tutorials he has written over the years.