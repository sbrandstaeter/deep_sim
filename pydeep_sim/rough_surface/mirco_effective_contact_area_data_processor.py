""" Utility classes needed for the data generation workflow"""

import logging

import numpy as np

from queens.data_processors.txt_file import TxtFile

_logger = logging.getLogger(__name__)


class MircoEffectiveContactAreaFromLogFile(TxtFile):
    def filter_and_manipulate_raw_data(self, raw_data):
        """Filter the raw data from the txt file.

        The DataProcessorTxt class provides some basic filtering functionality,
        however it is up to the user to define the specifics of how the raw data
        should be filtered.

        Args:
            raw_data (lst): List of strings Raw data from file.

        Return:
            To be implemented by user.
        """
        regex_global = r"Effective contact area fraction is: "
        regex_numeric_vals = r"\b\d+\.\d+\b"
        global_matches = self._extract_lines_with_regex(  # pylint: disable=W0212
            raw_data, regex_global
        )
        numeric_vals = [
            self._extract_quantities_from_line(  # pylint: disable=W0212
                global_match[1], regex_numeric_vals
            )
            for global_match in global_matches
        ]
        _logger.info("\nCurrent numeric vals:")
        _logger.info(str(numeric_vals))
        numeric_vals = np.array(numeric_vals, dtype="float64").ravel().tolist()
        if not numeric_vals:
            return None
        return numeric_vals


MIRCO_EFFECTIVE_CONTACT_AREA_DATAPROCESSOR = MircoEffectiveContactAreaFromLogFile(
    file_name_identifier="*.log",
    file_options_dict={},
    files_to_be_deleted_regex_lst=None,
    remove_logger_prefix_from_raw_data=False,
    max_file_size_in_mega_byte=200,
)
