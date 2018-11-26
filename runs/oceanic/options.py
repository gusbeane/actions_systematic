from configparser import ConfigParser


class options_reader(object):
    def __init__(self, file):
        self.parser = ConfigParser(inline_comment_prefixes=';')
        try:
            self.parser.read(file)
        except:
            print('couldnt read options file')
            raise Exception('Cant read the provided options file:' + file)

        self.options = {}

        # read in general parameters
        for opt in ['output_directory']:
            self._read_required_option_('general', opt)

        self._read_optional_option_('general', 'ncpu', '1')
        self._read_optional_option_('general', 'gpu_enabled', 'false')
        self._read_optional_option_('general', 'ngpu', '1')
        self._read_optional_option_('general', 'write_frequency', '10')
        self._read_optional_option_('general', 'out_file', 'cluster_snapshots.p')

        # read in simulation parameters
        for opt in ['simulation_directory', 'cache_directory', 'snap_index',
                    'star_softening_in_pc', 'dark_softening_in_pc']:
            self._read_required_option_('simulation', opt)

        self._read_optional_option_('simulation', 'sim_name', None)
        self._read_optional_option_('simulation', 'star_char_mass', None)
        self._read_optional_option_('simulation', 'dark_char_mass', None)
        self._read_optional_option_('simulation', 'gal_info', None)
        if self.options['star_char_mass'] is not None:
            self.options['star_char_mass'] = float(self.options['star_char_mass'])
        if self.options['dark_char_mass'] is not None:
            self.options['dark_char_mass'] = float(self.options['dark_char_mass'])

        # read in force_calculation parameters
        for opt in ['Rmax', 'theta', 'period']:
            self._read_required_option_('force_calculation', opt)

        self._read_optional_option_('force_calculation', 'axisymmetric', 'false')
        self.options['axisymmetric'] = self._convert_bool_(self.options['axisymmetric'])
        if self.options['axisymmetric']:
            self._read_optional_option_('force_calculation', 'axi_Rinit', None)
            self._read_optional_option_('force_calculation', 'axi_vcircfrac', None)
            self._read_optional_option_('force_calculation', 'axi_zinit', None)
            if self.options['axi_Rinit'] is not None and self.options['axi_vcircfrac'] is not None \
                and self.options['axi_zinit'] is not None:
                self.options['axi_Rinit'] = float(self.options['axi_Rinit'])
                self.options['axi_vcircfrac'] = float(self.options['axi_vcircfrac'])
                self.options['axi_zinit'] = float(self.options['axi_zinit'])

        # read in cluster parameters
        for opt in ['N', 'W0', 'Rcluster', 'softening',
                    'nbodycode', 'use_kroupa',
                    'timestep', 'tend']:
            self._read_required_option_('cluster', opt)

        self._read_optional_option_('cluster', 'eject_cut', '300.0')
        self._read_optional_option_('cluster', 'Mcluster', None)
        self._read_optional_option_('cluster', 'kroupa_max', '100.0')

        # read in starting_star parameters
        # for opt in ['ss_Rmin', 'ss_Rmax', 'ss_zmin', 'ss_zmax']:
        #    self._read_required_option_('starting_star', opt)

        self._read_optional_option_('starting_star', 'ss_Rmin', None)
        self._read_optional_option_('starting_star', 'ss_Rmax', None)
        self._read_optional_option_('starting_star', 'ss_zmin', None)
        self._read_optional_option_('starting_star', 'ss_zmax', None)

        self._read_optional_option_('starting_star', 'ss_agemin_in_Gyr', '0')
        self._read_optional_option_('starting_star', 'ss_agemax_in_Gyr', '0')
        self._read_optional_option_('starting_star', 'ss_seed', '1776')
        self._read_optional_option_('starting_star', 'ss_id', None)

        self._read_optional_option_('starting_star', 'ss_action_cuts', 'false')
        self.options['ss_action_cuts'] =\
            self._convert_bool_(self.options['ss_action_cuts'])
        if self.options['ss_action_cuts']:
            for opt in ['Jr_min', 'Jr_max', 'Jz_min', 'Jz_max']:
                self._read_required_option_('starting_star', opt)

        ss_array = [self.options['ss_Rmin'], self.options['ss_Rmax'],
                    self.options['ss_zmin'], self.options['ss_zmax']]

        if self.options['ss_id'] is None and None in ss_array:
            print('if ss_id is not given, you must provide Rmin, Rmax,')
            print('zmin, zmax. Exiting...')
            raise Exception('insufficient information to determine ss')
        elif self.options['ss_id'] is not None:
            self.options['ss_id'] = int(self.options['ss_id'])

        # convert relevant parameters
        int_options = ['snap_index',
                       'ss_seed', 'write_frequency',
                       'ncpu', 'ngpu', 'N', 'W0']
        for opt in int_options:
            if opt in self.options.keys():
                self.options[opt] = int(self.options[opt])

        float_options = ['ss_Rmin', 'ss_Rmax', 'ss_zmin', 'ss_zmax',
                         'ss_agemin_in_Gyr', 'ss_agemax_in_Gyr', 'Mcluster',
                         'Rcluster',
                         'softening', 'eject_cut', 'timestep', 'tend', 
                         'star_softening_in_pc', 'dark_softening_in_pc',
                         'Rmax', 'theta', 'period',
                         'Jr_min', 'Jr_max', 'Jz_min', 'Jz_max']
        for opt in float_options:
            if opt in self.options.keys():
                self.options[opt] = float(self.options[opt])

        bool_options = ['gpu_enabled', 'use_kroupa']
        for opt in bool_options:
            if opt in self.options.keys():
                self.options[opt] = self._convert_bool_(self.options[opt])

        self._convert_nbodycode_(self.options['nbodycode'])

    def set_options(self, object):
        for key in self.options.keys():
            setattr(object, key, self.options[key])

    def _convert_bool_(self, string):
        if string in ['True', 'true', '1', 1]:
            return True
        elif string in ['False', 'false', '0', 0]:
            return False
        else:
            raise Exception("Can't recognize bool:", string, "in options file")

    def _read_required_option_(self, category, option):
        try:
            self.options[option] = self.parser.get(category, option)
            print('set option: ', option, ' as: ', self.options[option])
        except:
            raise Exception('Couldnt find required option: ' + option)

    def _read_optional_option_(self, category, option, default):
        try:
            self.options[option] = self.parser.get(category, option)
            print('set option: ', option, ' as: ', self.options[option])
        except:
            print('Couldnt find option', option, ', using default: ', default)
            self.options[option] = default

    def _convert_nbodycode_(self, code_string):
        if code_string == 'ph4':
            from amuse.community.ph4.interface import ph4
            self.options['nbodycode'] = ph4
        else:
            raise Exception("Can't recognize given code: "+code_string)


if __name__ == '__main__':
    import sys
    g = options_reader(sys.argv[1])
