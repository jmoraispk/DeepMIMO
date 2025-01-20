"""
DeepMIMO dataset adapter for Sionna.

This module provides functionality to adapt DeepMIMO datasets for use with Sionna.
It handles:
- Channel data formatting and conversion
- Multi-user scenario support
- Multi-basestation scenario support

The adapter assumes BSs are transmitters and users are receivers. Uplink channels
can be generated using (transpose) reciprocity.
"""

# Standard library imports
from typing import Optional, List, Tuple, Any

# Third-party imports
import numpy as np
from numpy.typing import NDArray


class DeepMIMOSionnaAdapter:
    """Class for converting DeepMIMO dataset format to Sionna format.
    
    This class handles the conversion of channel data from DeepMIMO format to
    the format expected by Sionna, supporting various configurations of BSs and UEs.
    
    Attributes:
        dataset (dict): The loaded DeepMIMO dataset.
        bs_idx (NDArray): Array of basestation indices to include.
        ue_idx (NDArray): Array of user indices to include.
        num_rx_ant (int): Number of receiver antennas.
        num_tx_ant (int): Number of transmitter antennas.
        num_samples_bs (int): Number of basestation samples.
        num_samples_ue (int): Number of user samples.
        num_samples (int): Total number of channel samples.
        num_rx (int): Number of receivers per sample.
        num_tx (int): Number of transmitters per sample.
        num_paths (int): Number of paths per channel.
        num_time_steps (int): Number of time steps (1 for static).
        ch_shape (tuple): Required shape for channel coefficients.
        t_shape (tuple): Required shape for path delays.
    """

    def __init__(self, DeepMIMO_dataset: dict, bs_idx: Optional[int | List[int] | NDArray] = None,
                 ue_idx: Optional[int | List[int] | NDArray] = None) -> None:
        """Initialize the Sionna adapter.
        
        Args:
            DeepMIMO_dataset (dict): The loaded DeepMIMO dataset.
            bs_idx (Optional[int | List[int] | NDArray]): Basestation indices to include. Defaults to [0].
            ue_idx (Optional[int | List[int] | NDArray]): User indices to include. Defaults to all users.
            
        Examples:
            Multi-user channels:
            >>> ue_idx = np.array([[0, 1, 2], [1, 2, 3]])  # (num_bs x 3 UEs)
            
            Multi-BS channels:
            >>> bs_idx = np.array([[0, 1], [2, 3]])  # (2 BSs x num_rx)
        """
        self.dataset = DeepMIMO_dataset
        
        # Set bs_idx based on given parameters
        # If no input is given, choose the first basestation
        if bs_idx is None:
            bs_idx = np.array([[0]])
        self.bs_idx = self._verify_idx(bs_idx)
        
        # Set ue_idx based on given parameters
        # If no input is given, set all user indices
        if ue_idx is None:
            ue_idx = np.arange(DeepMIMO_dataset[0]['user']['channel'].shape[0])
        self.ue_idx = self._verify_idx(ue_idx)
        
        # Extract number of antennas from the DeepMIMO dataset
        self.num_rx_ant = DeepMIMO_dataset[0]['user']['channel'].shape[1]
        self.num_tx_ant = DeepMIMO_dataset[0]['user']['channel'].shape[2]
        
        # Determine the number of samples based on the given indices
        self.num_samples_bs = self.bs_idx.shape[0]
        self.num_samples_ue = self.ue_idx.shape[0]
        self.num_samples = self.num_samples_bs * self.num_samples_ue
        
        # Determine the number of tx and rx elements in each channel sample
        self.num_rx = self.ue_idx.shape[1]
        self.num_tx = self.bs_idx.shape[1]
        
        # Determine the number of available paths in the DeepMIMO dataset
        self.num_paths = DeepMIMO_dataset[0]['user']['channel'].shape[-1]
        self.num_time_steps = 1  # Time step = 1 for static scenarios
        
        # The required path power shape for Sionna
        self.ch_shape = (self.num_rx, self.num_rx_ant, self.num_tx, self.num_tx_ant, 
                        self.num_paths, self.num_time_steps)
        
        # The required path delay shape for Sionna
        self.t_shape = (self.num_rx, self.num_tx, self.num_paths)
    
    def _verify_idx(self, idx: int | List[int] | NDArray) -> NDArray:
        """Verify and format input indices.
        
        This function checks and converts input indices to the proper format,
        handling various input types and dimensions.
        
        Args:
            idx (int | List[int] | NDArray): Input indices in various formats.
            
        Returns:
            NDArray: Verified and formatted indices as numpy array.
            
        Raises:
            TypeError: If index input type is invalid.
            ValueError: If index dimensions are invalid.
        """
        idx = self._idx_to_numpy(idx)
        idx = self._numpy_size_check(idx)
        return idx
    
    def _idx_to_numpy(self, idx: int | List[int] | range | NDArray) -> NDArray:
        """Convert input indices to numpy array format.
        
        This function handles conversion of various input types to numpy arrays.
        
        Args:
            idx (int | List[int] | range | NDArray): Input indices as integer, list, range, or numpy array.
            
        Returns:
            NDArray: Input converted to numpy array.
            
        Raises:
            TypeError: If input type is not supported.
        """
        if isinstance(idx, int): 
            idx = np.array([[idx]])
        elif isinstance(idx, list) or isinstance(idx, range): 
            idx = np.array(idx)
        elif isinstance(idx, np.ndarray):
            pass
        else:
            raise TypeError('The index input type must be an integer, list, or numpy array!') 
        return idx
    
    def _numpy_size_check(self, idx: NDArray) -> NDArray:
        """Check and format numpy array dimensions.
        
        This function ensures numpy arrays have the correct dimensionality
        for channel index specifications.
        
        Args:
            idx (NDArray): Input numpy array.
            
        Returns:
            NDArray: Properly shaped numpy array.
            
        Raises:
            ValueError: If input dimensions are invalid.
        """
        if len(idx.shape) == 1:
            idx = idx.reshape((-1, 1))
        elif len(idx.shape) == 2:
            pass
        else:
            raise ValueError('The index input must be integer, vector or 2D matrix!')
        return idx
    
    def __len__(self) -> int:
        """Get number of available channel samples.
        
        Returns:
            int: Total number of channel samples.
        """
        return self.num_samples
        
    def __call__(self) -> Tuple[NDArray, NDArray]:
        """Generate channel samples in Sionna format.
        
        This function yields channel samples one at a time, converting them
        from DeepMIMO format to Sionna format.
        
        Returns:
            Tuple[NDArray, NDArray]: Tuple containing:
                - Channel coefficients array of shape ch_shape.
                - Path delays array of shape t_shape.
        """
        for i in range(self.num_samples_ue):  # For each UE sample
            for j in range(self.num_samples_bs):  # For each BS sample
                # Generate zero vectors for the Sionna sample
                a = np.zeros(self.ch_shape, dtype=np.csingle)
                tau = np.zeros(self.t_shape, dtype=np.single)
                
                # Place the DeepMIMO dataset power and delays into the channel sample
                for i_ch in range(self.num_rx):  # for each receiver in the sample
                    for j_ch in range(self.num_tx):  # for each transmitter in sample
                        i_ue = self.ue_idx[i][i_ch]  # UE channel sample i - RX i_ch
                        i_bs = self.bs_idx[j][j_ch]  # BS channel sample i - TX j_ch
                        a[i_ch, :, j_ch, :, :, 0] = self.dataset[i_bs]['user']['channel'][i_ue]
                        tau[i_ch, j_ch, :self.dataset[i_bs]['user']['paths'][i_ue]['num_paths']] = \
                            self.dataset[i_bs]['user']['paths'][i_ue]['ToA'] 
                
                yield (a, tau)  # yield this sample
