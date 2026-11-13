"""Crée un projet QGIS (.qgs) avec toutes les mosaïques d'un dossier de résultats.

Usage : python creer_projet_qgis.py "D:\\...\\ODM Santa Luzia" "Mosaics Santa Luzia"
Chaque sous-dossier « <nom>\\<nom>.tif » devient une couche raster RGB + alpha, avec un chemin relatif
au projet (le projet reste valable si le dossier est copié ailleurs ou si la lettre du disque change).
"""
import os, re, sys, uuid
from xml.sax.saxutils import escape, quoteattr
import rasterio

dossier, titre = sys.argv[1], sys.argv[2]

SRS = """<spatialrefsys nativeFormat="Wkt">
  <wkt>PROJCRS["WGS 84 / UTM zone 26N",BASEGEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]],CONVERSION["UTM zone 26N",METHOD["Transverse Mercator",ID["EPSG",9807]],PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],PARAMETER["Longitude of natural origin",-27,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],PARAMETER["Scale factor at natural origin",0.9996,SCALEUNIT["unity",1],ID["EPSG",8805]],PARAMETER["False easting",500000,LENGTHUNIT["metre",1],ID["EPSG",8806]],PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1]],ID["EPSG",32626]]</wkt>
  <proj4>+proj=utm +zone=26 +datum=WGS84 +units=m +no_defs</proj4>
  <srid>32626</srid><authid>EPSG:32626</authid>
  <description>WGS 84 / UTM zone 26N</description>
  <projectionacronym>utm</projectionacronym><ellipsoidacronym>EPSG:7030</ellipsoidacronym>
  <geographicflag>false</geographicflag>
</spatialrefsys>"""

def num(nom):
    m = re.search(r"(\d+)\s*$", nom)
    return int(m.group(1)) if m else 0

couches = []
for nom in sorted(os.listdir(dossier), key=num):
    tif = os.path.join(dossier, nom, nom + ".tif")
    if not os.path.isfile(tif):
        continue
    with rasterio.open(tif) as s:
        if s.crs.to_epsg() != 32626:
            print("ignoree (CRS different) :", tif); continue
        b = s.bounds
    couches.append(dict(nom=nom, id=re.sub(r"\W", "_", nom) + "_" + uuid.uuid4().hex[:8],
                        source="./" + nom + "/" + nom + ".tif", b=b))
if not couches:
    sys.exit("aucune mosaique trouvee")

xmin = min(c["b"].left for c in couches); xmax = max(c["b"].right for c in couches)
ymin = min(c["b"].bottom for c in couches); ymax = max(c["b"].top for c in couches)

def ext(b):
    return f"<extent><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin><xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></extent>"

def contraste(c):
    return (f"<{c}ContrastEnhancement><minValue>0</minValue><maxValue>255</maxValue>"
            f"<algorithm>NoEnhancement</algorithm></{c}ContrastEnhancement>")

arbre = "\n".join(
    f'      <layer-tree-layer id={quoteattr(c["id"])} name={quoteattr(c["nom"])} source={quoteattr(c["source"])} '
    f'providerKey="gdal" checked="Qt::Checked" expanded="0" legend_exp=""><customproperties/></layer-tree-layer>'
    for c in reversed(couches))   # le vol 1 en bas de la liste, comme dans QGIS

ordre = "\n".join(f'    <item>{escape(c["id"])}</item>' for c in reversed(couches))

maplayers = "\n".join(f"""  <maplayer type="raster" refreshOnNotifyEnabled="0" autoRefreshTime="0" hasScaleBasedVisibilityFlag="0" styleCategories="AllStyleCategories">
    {ext((c["b"].left, c["b"].bottom, c["b"].right, c["b"].top))}
    <id>{escape(c["id"])}</id>
    <datasource>{escape(c["source"])}</datasource>
    <layername>{escape(c["nom"])}</layername>
    <srs>{SRS}</srs>
    <provider>gdal</provider>
    <noData><noDataList bandNo="1" useSrcNoData="0"/><noDataList bandNo="2" useSrcNoData="0"/><noDataList bandNo="3" useSrcNoData="0"/><noDataList bandNo="4" useSrcNoData="0"/></noData>
    <pipe>
      <provider><resampling enabled="false" zoomedInResamplingMethod="nearestNeighbour" zoomedOutResamplingMethod="nearestNeighbour" maxOversampling="2"/></provider>
      <rasterrenderer type="multibandcolor" redBand="1" greenBand="2" blueBand="3" alphaBand="4" opacity="1" nodataColor="">
        <rasterTransparency/>
        <minMaxOrigin><limits>None</limits><extent>WholeRaster</extent><statAccuracy>Estimated</statAccuracy><cumulativeCutLower>0.02</cumulativeCutLower><cumulativeCutUpper>0.98</cumulativeCutUpper><stdDevFactor>2</stdDevFactor></minMaxOrigin>
        {contraste("red")}{contraste("green")}{contraste("blue")}
      </rasterrenderer>
      <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
      <huesaturation saturation="0" grayscaleMode="0" colorizeOn="0" colorizeRed="255" colorizeGreen="128" colorizeBlue="128" colorizeStrength="100" invertColors="0"/>
      <rasterresampler maxOversampling="2"/>
      <resamplingStage>resamplingFilter</resamplingStage>
    </pipe>
    <blendMode>0</blendMode>
  </maplayer>""" for c in couches)

qgs = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis projectname={quoteattr(titre)} version="3.34.0-Prizren">
  <homePath path=""/>
  <title>{escape(titre)}</title>
  <autotransaction active="0"/>
  <evaluateDefaultValues active="0"/>
  <trust active="0"/>
  <projectCrs>{SRS}</projectCrs>
  <layer-tree-group>
    <customproperties/>
    <layer-tree-group name={quoteattr(titre)} checked="Qt::Checked" expanded="1">
      <customproperties/>
{arbre}
    </layer-tree-group>
    <custom-order enabled="0">
{ordre}
    </custom-order>
  </layer-tree-group>
  <mapcanvas name="theMapCanvas" annotationsVisible="1">
    <units>meters</units>
    {ext((xmin, ymin, xmax, ymax))}
    <rotation>0</rotation>
    <destinationsrs>{SRS}</destinationsrs>
    <rendermaptile>0</rendermaptile>
  </mapcanvas>
  <projectlayers>
{maplayers}
  </projectlayers>
  <properties>
    <Paths><Absolute type="bool">false</Absolute></Paths>
    <Measurement><DistanceUnits type="QString">meters</DistanceUnits><AreaUnits type="QString">m2</AreaUnits></Measurement>
  </properties>
</qgis>
"""
sortie = os.path.join(dossier, titre + ".qgs")
open(sortie, "w", encoding="utf-8").write(qgs)
print(f"{len(couches)} mosaiques -> {sortie}")
for c in couches:
    print(f"  {c['nom']}: {c['b'].right-c['b'].left:.0f} x {c['b'].top-c['b'].bottom:.0f} m")
