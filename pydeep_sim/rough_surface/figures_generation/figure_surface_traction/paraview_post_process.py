from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

pmin = -0.03
pmax = +0.03
field = 'hv'

TEXTWIDTH = 507 #pt

# 96 px = 72 pt 
res_width = int(TEXTWIDTH*(96/72)/3) #px
res_height = int(TEXTWIDTH*(96/72)/3) #px

layout_width = res_width #px
layout_height = res_height #px

fontsize = 8
title = '$z/L$'

# create a new 'CSV Reader'
rough_traction_fieldcsv = CSVReader(registrationName='rough_traction_field.csv', FileName=['C:\\Users\\bona_ja\\Desktop\\figure_topographies_traction\\rough_traction_field.csv'])
rough_traction_fieldcsv.DetectNumericColumns = 1
rough_traction_fieldcsv.UseStringDelimiter = 1
rough_traction_fieldcsv.HaveHeaders = 1
rough_traction_fieldcsv.FieldDelimiterCharacters = ','
rough_traction_fieldcsv.AddTabFieldDelimiter = 0
rough_traction_fieldcsv.MergeConsecutiveDelimiters = 0

# create a new 'Table To Points'
tableToPoints1 = TableToPoints(registrationName='TableToPoints1', Input=rough_traction_fieldcsv)
tableToPoints1.XColumn = 'xv'
tableToPoints1.YColumn = 'yv'
tableToPoints1.ZColumn = field
tableToPoints1.a2DPoints = 1
tableToPoints1.KeepAllDataArrays = 1

# create a new 'Delaunay 2D'
delaunay2D1 = Delaunay2D(registrationName='Delaunay2D1', Input=tableToPoints1)
delaunay2D1.ProjectionPlaneMode = 'XY Plane'
delaunay2D1.Alpha = 0.0
delaunay2D1.Tolerance = 1e-05
delaunay2D1.Offset = 1.0
delaunay2D1.BoundingTriangulation = 0

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# Properties modified on renderView1
renderView1.OrientationAxesVisibility = 0

# set active source
SetActiveSource(delaunay2D1)

# get display properties
delaunay_trDisplay = GetDisplayProperties(delaunay2D1, view=renderView1)

# set scalar coloring
ColorBy(delaunay_trDisplay, ('POINTS', field))

# get 2D transfer function for field
tr_29TF2D = GetTransferFunction2D(field)
tr_29TF2D.AutomaticRescaleRangeMode = "Never"
tr_29TF2D.RescaleTransferFunction(pmin, pmax, 0.0, 1.0)

# get color transfer function/color map for field
tr_29LUT = GetColorTransferFunction(field)
tr_29LUT.AutomaticRescaleRangeMode = "Never"
tr_29LUT.RescaleTransferFunction(pmin,pmax)

# get opacity transfer function/opacity map for field
tr_29PWF = GetOpacityTransferFunction(field)
tr_29PWF.RescaleTransferFunction(pmin,pmax)

tr_29LUT.ApplyPreset('Cool to warm', True)

# update the view to ensure updated data information
renderView1.Update()

# show data in view
surfaceDisplay = Show(delaunay2D1, renderView1, 'GeometryRepresentation')

## Render all views to see them appears
RenderAllViews()

# get color legend/bar for tr_29LUT in view renderView1
tr_29LUTColorBar = GetScalarBar(tr_29LUT, renderView1)
tr_29LUTColorBar.Orientation = 'Horizontal'
tr_29LUTColorBar.WindowLocation = 'Any Location'
tr_29LUTColorBar.Title = title
tr_29LUTColorBar.ComponentTitle = ''
tr_29LUTColorBar.TitleColor = [0.0, 0.0, 0.0]
tr_29LUTColorBar.TitleFontFamily = 'Times'
tr_29LUTColorBar.TitleFontSize = fontsize+2
tr_29LUTColorBar.LabelColor = [0.0, 0.0, 0.0]
tr_29LUTColorBar.LabelFontFamily = 'Times'
tr_29LUTColorBar.LabelFontSize = fontsize
tr_29LUTColorBar.ScalarBarLength = 0.4
tr_29LUTColorBar.Position = [0.5-0.4/2, 0.85]
tr_29LUTColorBar.ScalarBarThickness = 8
tr_29LUTColorBar.RangeLabelFormat = '%-#3.3g'
tr_29LUTColorBar.UseCustomLabels = 1
tr_29LUTColorBar.CustomLabels = [1.0]

# Properties modified on renderView1.AxesGrid
renderView1.AxesGrid.AxesToLabel = 0
renderView1.AxesGrid.XTitleFontSize = fontsize
renderView1.AxesGrid.YTitleFontSize = fontsize
renderView1.AxesGrid.XLabelFontSize = fontsize
renderView1.AxesGrid.YLabelFontSize = fontsize
renderView1.AxesGrid.YTitle = 'y'
renderView1.AxesGrid.YAxisLabels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
renderView1.AxesGrid.XAxisLabels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# update the view to ensure updated data information
renderView1.Update()

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(layout_width,layout_height)

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.5, 0.6, 1.5]
renderView1.CameraFocalPoint = [0.5, 0.6, 1.0]
renderView1.CameraParallelScale = 0.7

# Render all views to see them appears
RenderAllViews()

# save screenshot
DPI = 300 # 1 inch = 96 px
mag_factor = 2*int(DPI/96)
SaveScreenshot(
    filename = r'C:\\Users\bona_ja\\Desktop\\figure_topographies_traction\\test_screenshot.png',
    viewOrLayout = renderView1,
    ImageResolution = (mag_factor*res_width,mag_factor*res_height))

# For latex: forcing  width = (1/mag_factor)\textwidth should result in mag_factor DPI
