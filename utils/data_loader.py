"""
This script processes grid and road emissions data for London boroughs, 
using shapefiles and Excel data to model atmospheric emissions.
"""

import numpy as np
import pandas as pd
import shapely
import geopandas as gpd

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

# file-paths
RAW_DATA_PATH = BASE_DIR / 'raw_data'
FILE_PATH = BASE_DIR / 'processed_data' / 'final_df.pkl'


RELEVANT_GRID_FEATURES = {
                        "Grid_ID_19": "grid_id",
                        "Borough": "borough",
                        "Zone": "zone",
                        "Area_km2": "area",
                        }

RELEVANT_ROAD_FEATURES = {
                        "Road Classification": "road_type",
                        " Speed (km/hr) - Except Buses ": "road_speed_non_bus",
                        " Speed (km/hr) - Buses Only ": "road_speed_bus",
                        " AADT Motorcycle ": "aadt_motorcycle",
                        " AADT Taxi ": "aadt_taxi",
                        " AADT Petrol Car ": "aadt_petrol_car",
                        " AADT Diesel Car ": "aadt_diesel_car",
                        " AADT Electric Car ": "aadt_electric_car",
                        " AADT Petrol PHV ": "aadt_petrol_phv",
                        " AADT Diesel PHV ": "aadt_diesel_phv",
                        " AADT Electric PHV ": "aadt_electric_phv",
                        " AADT Petrol LGV ": "aadt_petrol_lgv",
                        " AADT Diesel LGV ": "aadt_diesel_lgv",
                        " AADT Electric LGV ": "aadt_electric_lgv",
                        " AADT 2019 - HGVs - Rigid - 2 Axles ": "aadt_hgv_rigid_2_axles",
                        " AADT 2019 - HGVs - Rigid - 3 Axles ": "aadt_hgv_rigid_3_axles",
                        " AADT 2019 - HGVs - Rigid - 4 or more Axles ": "aadt_hgv_rigid_4_or_more_axles",
                        " AADT 2019 - HGVs - Articulated - 3 to 4 Axles ": "aadt_hgv_articulated_3_to_4_axles",
                        " AADT 2019 - HGVs - Articulated - 5 Axles ": "aadt_hgv_articulated_5_axles",
                        " AADT 2019 - HGVs - Articulated - 6 Axles ": "aadt_hgv_articulated_6_axles",
                        " AADT 2019 - Buses ": "aadt_bus",
                        " AADT 2019 - Coaches ": "aadt_coach",
                        " VKM Motorcycle ": "vkm_motorcycle",
                        " VKM Taxi ": "vkm_taxi",
                        " VKM Petrol Car ": "vkm_petrol_car",
                        " VKM Diesel Car ": "vkm_diesel_car",
                        " VKM Electric Car ": "vkm_electric_car",
                        " VKM Petrol PHV ": "vkm_petrol_phv",
                        " VKM Diesel PHV ": "vkm_diesel_phv",
                        " VKM Electric PHV ": "vkm_electric_phv",
                        " VKM Petrol LGV ": "vkm_petrol_lgv",
                        " VKM Diesel LGV ": "vkm_diesel_lgv",
                        " VKM Electric LGV ": "vkm_electric_lgv",
                        " VKM 2019 - HGVs - Rigid - 2 Axles ": "vkm_hgv_rigid_2_axles",
                        " VKM 2019 - HGVs - Rigid - 3 Axles ": "vkm_hgv_rigid_3_axles",
                        " VKM 2019 - HGVs - Rigid - 4 or more Axles ": "vkm_hgv_rigid_4_or_more_axles",
                        " VKM 2019 - HGVs - Articulated - 3 to 4 Axles ": "vkm_hgv_articulated_3_to_4_axles",
                        " VKM 2019 - HGVs - Articulated - 5 Axles ": "vkm_hgv_articulated_5_axles",
                        " VKM 2019 - HGVs - Articulated - 6 Axles ": "vkm_hgv_articulated_6_axles",
                        " VKM 2019 - Buses ": "vkm_bus",
                        " VKM 2019 - Coaches ": "vkm_coach",
                        }

def read_excel_into_df_with_specific_columns(file_path, columns={}):
    df = pd.read_excel(file_path, usecols=columns.keys())
    df.rename(columns=columns, inplace=True)
    return df

def read_shape_into_gdf_with_specific_columns(file_path, columns={}):
    gdf = gpd.read_file(file_path)
    geometry_name = gdf.geometry.name
    if geometry_name not in columns:
        columns[geometry_name] = geometry_name
    gdf = gdf[columns.keys()]
    gdf.rename(columns=columns, inplace=True)
    return gdf

class BreakOutError(Exception):
    pass

class GridEmissions:
    def __init__(self):
        grid_shape_path = RAW_DATA_PATH / "supporting_data/grid/LAEI2019_Grid.shp"
        self.df = read_shape_into_gdf_with_specific_columns(grid_shape_path, columns=RELEVANT_GRID_FEATURES)

    def _bridge_multiline(self, multiline, tolerance):
        multiline_list = list(multiline.geoms)
        start_again = True
        while start_again:
            try:
                start_again = False
                length = len(multiline_list)
                if length == 1:
                    return multiline_list[0]
                for i in range(0, length):
                    for j in range(i+1, length):
                        line1 = multiline_list[i]
                        line2 = multiline_list[j]

                        start1, end1 = line1.coords[0], line1.coords[-1]
                        start2, end2 = line2.coords[0], line2.coords[-1]

                        best_fit = None
                        line_length = tolerance
                        potentials = [
                                        shapely.geometry.LineString([end1, start2]),
                                        shapely.geometry.LineString([end1, end2]),
                                        shapely.geometry.LineString([start1, start2]),
                                        shapely.geometry.LineString([start1, end2]),
                                    ]
                        
                        for potential in potentials:
                            if (potential.length < line_length):
                                best_fit = potential
                                line_length = potential.length
                        if best_fit is not None:
                            del multiline_list[i]
                            del multiline_list[j-1]
                            multiline_list.append(shapely.ops.linemerge((line1, line2, best_fit)))
                            start_again = True
                            raise BreakOutError
            except BreakOutError:
                pass
        return multiline

    def _process_links_groupby(self, gdf, retain_fields=[], non_contiguity_tolerance=200):
        crs = gdf.crs

        exploded_gdf = gdf.explode(index_parts=True)
        merged_geom = shapely.ops.linemerge(exploded_gdf.geometry.tolist())

        if isinstance(merged_geom, shapely.geometry.MultiLineString):
            if non_contiguity_tolerance:
                merged_geom = self._bridge_multiline(merged_geom, non_contiguity_tolerance)
                # if not isinstance(merged_geom, shapely.geometry.MultiLineString):
                #     print("Fixed contiguity.")
                # else:
                #     print("Failed to fix contiguity.")

        result = gpd.GeoDataFrame({"geometry": [merged_geom]}, crs=crs)

        for field in retain_fields:
            result[field] = gdf[field].iloc[0]

        return result
    
    def _find_total_intersecting_agg_by_grid(self, df2, agg):
        df2 = df2.to_crs(self.df.crs)
        overlap = gpd.overlay(df1=self.df, df2=df2, how="intersection", keep_geom_type=False)
        aggregated_data = overlap.groupby("grid_id").agg(**agg).reset_index()

        self.df = self.df.merge(aggregated_data, how='left', on='grid_id')
        for column in agg.keys():
            self.df[column] = self.df[column].fillna(0.0).astype(float).apply(lambda x: round(x, 6))
    
    def _add_link_data(self, link_type, rel_path, link_id_col, retain_fields=[]):
        file_path = RAW_DATA_PATH / rel_path
        gdf = gpd.read_file(file_path, columns=None)

        compiled_gdf = gdf.groupby(link_id_col).apply(self._process_links_groupby, include_groups=False, retain_fields=retain_fields).reset_index(drop=True)

        agg_length_by_link = {f"total_{link_type}_length": ("geometry", lambda x: np.sum(x.length))}
        self._find_total_intersecting_agg_by_grid(compiled_gdf, agg_length_by_link)

        start_points = compiled_gdf.geometry.apply(lambda x: shapely.geometry.Point(x.coords[0]))  # Start point
        end_points = compiled_gdf.geometry.apply(lambda x: shapely.geometry.Point(x.coords[-1])) # End point

        all_points = pd.concat([start_points, end_points])
        all_points_counts = all_points.value_counts()

        all_points_gdf = gpd.GeoDataFrame(geometry=all_points)
        all_points_gdf[f"total_{link_type}_link_terminations"] = all_points_gdf.geometry.map(all_points_counts)

        all_points_gdf = all_points_gdf.drop_duplicates(subset="geometry").reset_index(drop=True)

        agg_stations_by_link = {f"total_{link_type}_stations": ("geometry", np.size),
                               f"total_{link_type}_link_terminations": (f"total_{link_type}_link_terminations", np.sum)}

        self._find_total_intersecting_agg_by_grid(all_points_gdf, agg_stations_by_link)


    def add_emissions_data(self):
        emissions_summary_path = RAW_DATA_PATH / "emissions_summary"
        in_scope_dfs = [self.df]
        for path in emissions_summary_path.rglob("*-all-sources.shp"):
            print(f"Loading file {str(path).split("/")[-1]}")

            emissions_df = gpd.read_file(path,
                                        columns=("pollutant", "all_2019"))

            pollutant = emissions_df.iloc[0]["pollutant"]
            emissions_df.rename(columns={"all_2019": f"total_pollutant_{pollutant}_2019"},
                                inplace=True)

            in_scope_dfs.append(emissions_df.drop(columns=["pollutant", "geometry"]))
        self.df = pd.concat(in_scope_dfs, axis=1)

    def add_rail_data(self):
        self._add_link_data("rail", "supporting_data/rail/LAEI_Rail_Network_Link_ExactCut.shp", "LinkID", retain_fields=["DESCRIPT"])

    def add_shipping_data(self):
        self._add_link_data("shipping", "supporting_data/shipping/LAEI_PassengerShipping_Network_Link.shp", "LINK_ID")
    
    def add_road_data(self):
        shape_file_path = RAW_DATA_PATH / "supporting_data/road/shape/laei-2019-major-roads-final-unique-toids-flows-speeds-osgb.shp"
        gdf = read_shape_into_gdf_with_specific_columns(shape_file_path, columns={})


        excel_file_path = RAW_DATA_PATH / "supporting_data/road/excel/laei-2019-major-roads-vkm-flows-speeds.xlsx"
        df = read_excel_into_df_with_specific_columns(excel_file_path, columns=RELEVANT_ROAD_FEATURES)

        gdf = gdf.merge(df, left_index=True, right_index=True)
        non_numeric = ("geometry", "road_type")
        numeric_cols = [col for col in gdf.columns if col not in non_numeric]
        gdf[numeric_cols] = gdf[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        gdf = gdf.to_crs(self.df.crs)

        road_type_lookup = {"A": "a_road",
                            "B": "b_road",
                            "C": "c_unclassified_road",
                            "M": "motorway"}
        gdf["road_type"] = gdf["road_type"].apply(lambda x: "motorway" if x == "A1M" else road_type_lookup[x[0]])

        gdf["toid_road_length"] = gdf.geometry.length

        temp_df = self.df[["grid_id"]]

        overlap = gpd.overlay(df1=self.df, df2=gdf, how="intersection", keep_geom_type=False)
        overlap['length'] = overlap.geometry.length
        overlap['ratio'] = overlap["length"] / overlap["toid_road_length"]
        vkm_cols = [col for col in overlap.columns if col.startswith('vkm_')]
        for col in vkm_cols:
            overlap[col] = overlap[col] * overlap['ratio']
        aadt_cols = [col for col in overlap.columns if col.startswith('aadt_')]
        overlap_by_grid_id = overlap.groupby("grid_id")
        vkm_sums = overlap_by_grid_id[vkm_cols].sum()
        aadt_sums = overlap_by_grid_id[aadt_cols].sum()
        temp_df = temp_df.merge(vkm_sums, how='left', left_on='grid_id', right_index=True)
        temp_df = temp_df.merge(aadt_sums, how='left', left_on='grid_id', right_index=True)


        road_grouping = overlap.groupby(["grid_id", "road_type"])

        road_lengths = road_grouping["length"].sum().reset_index()
        road_lengths_pivot = road_lengths.pivot(index="grid_id", columns="road_type", values="length")
        road_lengths_pivot.columns = [f'total_road_length_{col}' for col in road_lengths_pivot.columns]
        temp_df = temp_df.merge(road_lengths_pivot, how='left', left_on='grid_id', right_index=True)

        road_counts = road_grouping.size().reset_index(name="road_count")
        road_counts_pivot = road_counts.pivot(index="grid_id", columns="road_type", values="road_count")
        road_counts_pivot.columns = [f"total_road_count_{col}" for col in road_counts_pivot.columns]
        temp_df = temp_df.merge(road_counts_pivot, how="left", left_on="grid_id", right_index=True)

        aggregated_data = overlap.groupby("grid_id").agg(**{"mean_road_speed_non_bus": ("road_speed_non_bus", "mean"),
                                        "mean_road_speed_bus": ("road_speed_bus", "mean")})
        temp_df = temp_df.merge(aggregated_data, how='left', on='grid_id')

        temp_df.fillna(value=0, inplace=True)
        self.df = self.df.merge(temp_df, how='left', on='grid_id')

    def encode_categorial_data(self, one_hot=[], ordinal={}):
        for col in one_hot:
            one_hot_encoded = pd.get_dummies(self.df[col], prefix=col)
            self.df = pd.concat([self.df, one_hot_encoded], axis=1)
        self.df.drop(columns=one_hot, inplace=True)

        for col, order in ordinal.items():
            ordinal_mapping = {category: index for index, category in enumerate(order)}
            self.df[col] = self.df[col].map(ordinal_mapping)



def load_and_save_data():
    # grid data object
    grid = GridEmissions()
    
    #adding emissions data sources to the grid data
    grid.add_emissions_data()  # core pollutant emissions data
    grid.add_rail_data()        # emissions data from rail sources
    grid.add_shipping_data()    # emissions data from shipping sources
    grid.add_road_data()        # emissions data from road sources
    
    #categorical data encoding for modeling
    grid.encode_categorial_data(
        one_hot=["borough"],  # one-hot encoding to 'borough' column
        ordinal={"zone": ("NonGLA", "OuterULEX", "InnerULEX", "Central")}  # ordinal encoding to 'zone'
    )
    
    #removing the geometry column as its not needed for further processing
    grid.df.drop(columns="geometry", inplace=True)
    
    #saving the processed df
    grid.df.to_pickle(FILE_PATH)
    print("Data saved successfully!")
    


if __name__ == "__main__":
    load_and_save_data()