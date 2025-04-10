# Dependencies

- plyfile
- lxml
- utm

Notes:
- One good format for conversions from blender is DAE (Collada). 
  We can manually extract data from OSM to blender and export to .DAE with 
  [this blender extension](https://github.com/gregdavisd/DAEBlend). 


# How the pipeline mechanism works:

1. Get map
2. Get TX-RX positions
3. Ray trace TX-RX positions on the map

# How each pipeline ray traces?

## Wireless InSite Pipeline

Creates the files necessary to call the insite executor.

## Sionna Pipeline

...





## Documentation of differences between Wireless InSite v3.x and v4

These changes apply only to the XML.
There are no changes anywhere else in terms of file generation.

1) In <APGAccelerationParameters>:
   1.1) delete tag <BinaryOutputRate>
   1.2) delete tag <WriteDB>
2) In <OutputRequest>:
   2.1) add tag <MaximumPermissibleExposure> (ideally, right after <H_FieldRMS>):
            <MaximumPermissibleExposure>
              <remcom::rxapi::OutputRequest>
                <Requested>
                  <remcom::rxapi::Boolean Value="false"/>
                </Requested>
                <ValidForModel>
                  <remcom::rxapi::Boolean Value="true"/>
                </ValidForModel>
              </remcom::rxapi::OutputRequest>
            </MaximumPermissibleExposure>
   2.2) add tag <PathStorageForRerun> (ideally, right after <PathLoss>):
            <PathStorageForRerun>
              <remcom::rxapi::OutputRequest>
                <Requested>
                  <remcom::rxapi::Boolean Value="false"/>
                </Requested>
                <ValidForModel>
                  <remcom::rxapi::Boolean Value="true"/>
                </ValidForModel>
              </remcom::rxapi::OutputRequest>
            </PathStorageForRerun>
   2.3) add tag <SParameter> (ideally, right after <ReceivedPower>):
            <SParameter>
              <remcom::rxapi::OutputRequest>
                <Requested>
                  <remcom::rxapi::Boolean Value="false"/>
                </Requested>
                <ValidForModel>
                  <remcom::rxapi::Boolean Value="false"/>
                </ValidForModel>
              </remcom::rxapi::OutputRequest>
            </SParameter>
3) In <remcom::rxapi::X3D>:
   3.1) delete tag <TerminalRefraction>
4) In <MaterialList>, for each <Material>:
   4.1) add tag <Volumetric> as below:
                            <Volumetric>
                              <remcom::rxapi::Boolean Value="false"/>
                            </Volumetric>
        NOTE: this is not necessary because the material is written directly to the
        .city and .ter files. Therefore, this tag is not needed. 
        It can even be deleted from the original templates.
5) In Antenna? Receiver or Transmitter tag (inside TxrxSetList)
  5.1) Delete tag <AntennaRotations>
  5.2) Add tag <AntennaAlignment> as below:
                    <AntennaAlignment>
                      <remcom::rxapi::SphericalAlignment>
                        <Phi>
                          <remcom::rxapi::Double Value="0"/>
                        </Phi>
                        <Roll>
                          <remcom::rxapi::Double Value="0"/>
                        </Roll>
                        <Theta>
                          <remcom::rxapi::Double Value="90"/>
                        </Theta>
                      </remcom::rxapi::SphericalAlignment>
                    </AntennaAlignment>
6) Add <Ellipsoid> to <remcom::rxapi::Scene>
        <Ellipsoid>
          <remcom::rxapi::EllipsoidEnum Value="EarthWGS84"/>
        </Ellipsoid>


## Verifying Wireless InSite pipeline results

The .txrx, .setup, .city, .ter, can be loaded into the UI with right-click -> Import (or Open)
Some parameters are *not* in these files that need to be changed in the UI, this is the case for:
- Carrier frequency: Needs to be changed in the waveforms tab for all automatically added waveforms;
- Antenna type: The default is half-wave dipole, but needs to be changed to Isotropic, by replacement or other methods. 

That's it. Then results should match either perfectly, or very well, i.e. less than 1% of points may have less than 0.3 dB pathloss variation - we are yet to discover what causes this slight difference. 