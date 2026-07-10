# Custom Imports
from observation_classes.combined_observation import CombinedObservation

# General Imports
import pandas as pd
from pathlib import Path



def complete_report(city_shortlist: dict, output_folder_path: str, sqkm = 500):
    dest_folder = Path(output_folder_path)
    summary_path = dest_folder / '0_summary.csv'
    summary_rows = []

    for city_name, city_id in city_shortlist.items():
        print(f'\n\nAttempting Collection for {city_name}')

        try:
            # Get observation
            obs = CombinedObservation(city_id, 
                                      sqkm=sqkm, 
                                      set_imgs=False,
                                      incl_stats_figure=False
                                      )

            # Collect Metadata
            file_name = str(f'{city_name}.pkl')
            dest_path = dest_folder / file_name
            metadata = obs.get_metadata()
            metadata['path'] = dest_path

            # Save observation
            obs.save(dest_path)

            # Checkpoint metadata
            summary_rows.append(metadata)
            summary = pd.DataFrame(summary_rows)
            summary.to_csv(summary_path, index=False)

        except Exception as e: 
            print(f'Could not create observation for {city_name}: {e}')


