import React from 'react';
import { Link } from 'react-router-dom';
import SearchIcon from '@mui/icons-material/Search';

const Navbar: React.FC = () => {
  return (
    <nav className="bg-white shadow-md p-4">
      <div className="container mx-auto flex justify-between items-center">
        <div className="text-xl font-bold text-gray-800">
          <Link to="/">RentPredict</Link>
        </div>
        <div className="space-x-4">
          <Link to="/try" className="text-gray-600 hover:text-gray-900 font-medium">Try</Link>
          <Link to="/about" className="text-gray-600 hover:text-gray-900 font-medium">About</Link>
          <Link to="/api" className="text-gray-600 hover:text-gray-900 font-medium">API</Link>
          <Link to="/debug" className="text-gray-600 hover:text-gray-900 font-medium">Debug</Link>
          <Link to="/login" className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-md shadow-lg shadow-blue-500/50 transition-shadow duration-300 ease-in-out hover:shadow-xl hover:shadow-blue-500/80">Login</Link>
          <Link to="/settings" className="text-gray-600 hover:text-gray-900 font-medium">Settings</Link>
          <Link to="/search" className="p-2 rounded-full hover:bg-gray-200">
            <SearchIcon />
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;