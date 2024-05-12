# PaddockTS  
Paddock-level phenology visualizations and spatiotemporal analyses using (for now) Sentinel2 data
  
### Initialization
Things you must do before the code will run:

### Note for JupyterLab setup on GADI
Include these options on the launch site:
- Storage: gdata/<my project>+gdata/v10
- Module Directories: /g/data/v10/public/modules/modulefiles/
- Modules: dea/20231204

### Programs
1. phenology videos
    - DONE
2. calendar plots
    - DONE mostly, needing some updates:
       - mark when interpolated
       - incorporate planetscope data?
3. paddock segmentation and clustering.
   - DONE
4. paddock quick-look flipbook
   - To-do
   - Base on paddock time series, but also get more spectral indices and band data. 
5. 'Feature learning'
   - To-do
   - This is where we exctract from paddock-level ts data the parameters that will feed into DAESIM e.g.
       - start/end of growth periods and other important transitions
       - predicted harvest date
       - predicted flowering

