import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import Try from './pages/Try';
import About from './pages/About';
import UserSettings  from './pages/UserSettings';
import Api from './pages/Api';
import Login from './pages/Login';
import Signup from './pages/Signup';
import ForgotPassword from './pages/ForgotPassword';
import ResetPassword from './pages/ResetPassword';
import Debug from './pages/Debug';
import RentChart from './pages/RentChart';
import Verify from './pages/Verify';
import Home from './pages/Home';
import CreateReport from './pages/CreateReport';
import ViewReport from './pages/ViewReport';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { Search } from '@mui/icons-material';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
  components: {
    MuiTextField: {
      styleOverrides: {
        root: {
          input: {
            "&:-webkit-autofill": {
              WebkitBoxShadow: "0 0 0 100px #E0D98C <<<<(Your color here) inset",
              WebkitTextFillColor: "default",
            },
          },
        },
      },
    }
  }
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      {/* CssBaseline is a helpful addition to normalize styles across browsers */}
      <CssBaseline />
      <Router>
        <Navbar />
        <div className="container mx-auto p-4">
          <Routes>
            <Route path="/try" element={<Try />} />
            <Route path="/about" element={<About />} />
            <Route path="/api" element={<Api />} />
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />
            <Route path="/forgot-password" element={<ForgotPassword />} />
            <Route path="/reset-password" element={<ResetPassword />} />
            <Route path="/verify" element={<Verify />} />
            <Route path="/debug" element={<Debug />} />
            <Route path="/rentchart" element={<RentChart />} />
            <Route path="/home" element={<Home />} />
            <Route path="/createreport" element={<CreateReport />} />
            <Route path="/viewreport" element={<ViewReport />} />
            <Route path="/settings" element={<UserSettings />} />
            <Route path="/search" element={<Search />} />
            <Route path="/" element={<Try />} />
          </Routes>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;