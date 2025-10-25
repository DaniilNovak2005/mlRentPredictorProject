const express = require('express');
const axios = require('axios');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const nodemailer = require('nodemailer');
const crypto = require('crypto');
const User = require('./models/User');
const Verification = require('./models/Verification');
const Report = require('./models/Report');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 3000;

// Auth Middleware
const auth = (req, res, next) => {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    if (!token) {
        return res.status(401).json({ msg: 'No token, authorization denied' });
    }
    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        req.user = decoded.user;
        next();
    } catch (err) {
        res.status(401).json({ msg: 'Token is not valid' });
    }
};

// Middleware
app.use(express.json());

// DB Connection
mongoose.connect(process.env.MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
})
.then(() => console.log('MongoDB connected'))
.catch(err => console.log(err));

// Nodemailer transporter
const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS
    }
});

app.get('/', (req, res) => {
    res.send('Rent Predictor Backend is running!');
});

// Signup
app.post('/signup', async (req, res) => {
    const { username, email, password } = req.body;

    console.log(`[${new Date().toISOString()}] User signup attempt for email: ${email}, username: ${username}`);

    try {
        let user = await User.findOne({ username: username });
        if (user) {
            return res.status(400).json({ msg: 'User already exists' });
        }
        user = await User.findOne({ email: email });
        if (user) {
            if (user.isVerified){
                return res.status(400).json({ msg: 'User already exists' });
            }
            await User.deleteOne({email: email})
        }

        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(password, salt);

        const newUser = new User({
            username,
            email,
            password: hashedPassword
        });

        await newUser.save();

        const verificationToken = crypto.randomBytes(32).toString('hex');
        const verification = new Verification({
            userId: newUser._id,
            token: verificationToken,
            isForAccountCreation: true
        });
        await verification.save();

        const verificationUrl = `${process.env.CLIENT_URL || 'http://localhost:5173'}/verify?token=${verificationToken}`;

        await transporter.sendMail({
            to: newUser.email,
            subject: 'Verify Your Account',
            html: `<div style="text-align: center;">
                     <h2>Welcome to Rent Predictor!</h2>
                     <p>Please verify your account by clicking the button below:</p>
                     <a href="${verificationUrl}" style="background-color: #3b82f6; color: white; font-weight: bold; padding: 14px 25px; text-align: center; text-decoration: none; display: inline-block; border-radius: 0.25rem;">Verify Account</a>
                   </div>`
        });

        res.status(201).json({ message: 'User created. Please check your email for verification link.' });

    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server error');
    }
});

// Login
app.post('/login', async (req, res) => {
    const { username, password } = req.body;

    try {
        let user = await User.findOne({ username: username });
        if (!user) {
            return res.status(400).json({ msg: 'Invalid credentials' });
        }

        if (!user.isVerified) {
            return res.status(400).json({ msg: 'Please verify your email to login' });
        }

        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return res.status(400).json({ msg: 'Invalid credentials' });
        }

        const payload = {
            user: {
                id: user.id
            }
        };

        jwt.sign(payload, process.env.JWT_SECRET, {
            expiresIn: 360000
        }, (err, token) => {
            if (err) throw err;
            res.json({ token });
        });

    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server error');
    }
});

app.get('/verify/:token', async (req, res) => {
    try {
        const verification = await Verification.findOne({ token: req.params.token });
        if (!verification) return res.status(400).json({ message: 'Invalid token' });

        const user = await User.findById(verification.userId);
        if (!user) return res.status(400).json({ message: 'User not found' });

        user.isVerified = true;
        await user.save();
        await verification.deleteOne();

        res.json({ message: 'Email verified successfully! You can now login.' });
    } catch (err) {
        console.error(err.message);
        res.status(500).json({ error: 'Server error' });
    }
});

// Forgot Password
app.post('/forgot-password', async (req, res) => {
    const { email } = req.body;

    try {
        const user = await User.findOne({ email });
        if (!user) {
            return res.status(400).json({ msg: 'User with that email does not exist' });
        }
        if (!user.isVerified) {
            return res.status(400).json({ msg: 'Please verify your account first' });
        }

        const verificationToken = crypto.randomBytes(32).toString('hex');
        const verification = new Verification({
            userId: user._id,
            token: verificationToken,
            isForAccountCreation: false
        });
        await verification.save();

        const verificationUrl = `${process.env.CLIENT_URL || 'http://localhost:5173'}/reset-password?token=${verificationToken}`;

        await transporter.sendMail({
            to: user.email,
            subject: 'Password Reset',
            html: `<div style="text-align: center;">
                     <h2>Password Reset Request</h2>
                     <p>Please reset your password by clicking the button below:</p>
                     <a href="${verificationUrl}" style="background-color: #3b82f6; color: white; font-weight: bold; padding: 14px 25px; text-align: center; text-decoration: none; display: inline-block; border-radius: 0.25rem;">Reset Password</a>
                   </div>`
        });

        res.status(200).json({ message: 'Password reset link sent to your email.' });

    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server error');
    }
});

// Reset Password
app.post('/reset-password/:token', async (req, res) => {
    const { password } = req.body;

    try {
        const verification = await Verification.findOne({ token: req.params.token });
        if (!verification) return res.status(400).send('Invalid token');

        const user = await User.findById(verification.userId);
        if (!user) return res.status(400).send('User not found');

        const salt = await bcrypt.genSalt(10);
        user.password = await bcrypt.hash(password, salt);

        await user.save();
        await verification.deleteOne();

        res.status(200).json({ message: 'Password has been reset.' });

    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server error');
    }
});

// Get User History
app.get('/history', auth, async (req, res) => {
    try {
        const user = await User.findById(req.user.id).select('history');
        if (!user) return res.status(404).json({ msg: 'User not found' });

        res.json({ history: user.history });
    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server error');
    }
});

// Get User Settings
app.get('/get-settings', auth, async (req, res) => {
    try {
        const user = await User.findById(req.user.id).select('settings');
        if (!user) return res.status(404).json({ msg: 'User not found' });

        let settings = user.settings;
        if (settings && settings.propertyAddresses) {
            try {
                const algorithm = 'aes-256-cbc';
                const encryptionKey = process.env.ENCRYPTION_KEY;
                if (!encryptionKey) {
                    return res.status(500).send('ENCRYPTION_KEY not set');
                }
                const key = Buffer.from(encryptionKey, 'hex');
                const iv = settings.propertyAddresses.iv ? Buffer.from(settings.propertyAddresses.iv, 'hex') : null;
                const encryptedData = settings.propertyAddresses.encryptedData ? Buffer.from(settings.propertyAddresses.encryptedData, 'hex') : null;

                if (iv && encryptedData) {
                    const decipher = crypto.createDecipheriv(algorithm, key, iv);
                    let decrypted = decipher.update(encryptedData);
                    decrypted = Buffer.concat([decrypted, decipher.final()]);
                    settings.propertyAddresses = JSON.parse(decrypted.toString());
                } else {
                    settings.propertyAddresses = [];
                }
            } catch (e) {
                console.error("Decryption error:", e);
                settings.propertyAddresses = [];
            }
        }

        res.json({ settings: settings });
    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server error');
    }
});

// Save User Settings
app.post('/save-settings', auth, async (req, res) => {
    try {
        const user = await User.findById(req.user.id);
        if (!user) return res.status(404).json({ msg: 'User not found' });

        const { settings, propertyAddresses } = req.body;

        // Encrypt property addresses if provided
        if (propertyAddresses) {
            const algorithm = 'aes-256-cbc';
            const encryptionKey = process.env.ENCRYPTION_KEY;
            if (!encryptionKey) {
                return res.status(500).send('ENCRYPTION_KEY not set');
            }
            const key = Buffer.from(encryptionKey, 'hex');
            const iv = crypto.randomBytes(16);

            const cipher = crypto.createCipheriv(algorithm, Buffer.from(key), iv);
            let encrypted = cipher.update(JSON.stringify(propertyAddresses));
            encrypted = Buffer.concat([encrypted, cipher.final()]);

            const encryptedData = {
                iv: iv.toString('hex'),
                encryptedData: encrypted.toString('hex')
            };

            settings.propertyAddresses = encryptedData;
        }

        user.settings = settings;
        await user.save();

        res.json({ message: 'Settings saved successfully' });
    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server error');
    }
});

// Update User History (Optional addition for adding entries)
app.post('/api/history', auth, async (req, res) => {
    const { entry } = req.body;

    try {
        const user = await User.findById(req.user.id);
        if (!user) return res.status(404).json({ msg: 'User not found' });

        user.history.push(entry);
        await user.save();

        res.json({ history: user.history });
    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server error');
    }
});

const getStateFromDetails = (details) => {
    // For now, this is a stub. Later, we can implement logic to determine state from details.
    if (details && details.State) {
        return details.State;
    }
    return 'FL'; // Default state
};

app.post('/predict-rent', async (req, res) => {
    const { zillowlink, details } = req.body;

    if (!zillowlink && !details) {
        return res.status(400).json({ error: 'Please provide features (details) or a Zillow link' });
    }

    const state = getStateFromDetails(details);
    const model = `xgb_model_${state}`;

    let payload;
    if (zillowlink) {
        payload = { model, link: zillowlink, state: state };
    } else {
        payload = { model, details: details, state: state };
    }

    try {
        const response = await axios.post('http://127.0.0.1:5000/predict', payload);
        res.json(response.data);
    } catch (error) {
        console.error('Error calling prediction service:', error.message);
        if (error.response) {
            res.status(error.response.status).json({ error: 'Error from prediction service' });
        } else if (error.request) {
            res.status(503).json({ error: 'Prediction service is unavailable' });
        } else {
            res.status(500).json({ error: 'Internal server error' });
        }
    }
});

app.post('/create-report-init', auth, async (req, res) => {
    const { zillowlink, details } = req.body;
    const userId = req.user.id;

    console.log(`[${new Date().toISOString()}] User ${userId} calling /create-report-init`);
    console.log(`[${new Date().toISOString()}] Request data:`, { zillowlink: zillowlink ? true : false, details: details ? true : false });

    if (!zillowlink && !details) {
        console.log(`[${new Date().toISOString()}] User ${userId}: Invalid request - no zillowlink or details provided`);
        return res.status(400).json({ error: 'Please provide features (details) or a Zillow link' });
    }
    const user = await User.findById(userId).select('settings');
    let settings = user.settings;
        if (settings && settings.propertyAddresses) {
            try {
                const algorithm = 'aes-256-cbc';
                const encryptionKey = process.env.ENCRYPTION_KEY;
                if (!encryptionKey) {
                    return res.status(500).send('ENCRYPTION_KEY not set');
                }
                const key = Buffer.from(encryptionKey, 'hex');
                const iv = settings.propertyAddresses.iv ? Buffer.from(settings.propertyAddresses.iv, 'hex') : null;
                const encryptedData = settings.propertyAddresses.encryptedData ? Buffer.from(settings.propertyAddresses.encryptedData, 'hex') : null;

                if (iv && encryptedData) {
                    const decipher = crypto.createDecipheriv(algorithm, key, iv);
                    let decrypted = decipher.update(encryptedData);
                    decrypted = Buffer.concat([decrypted, decipher.final()]);
                    settings.propertyAddresses = JSON.parse(decrypted.toString());
                } else {
                    settings.propertyAddresses = [];
                }
            } catch (e) {
                console.error("Decryption error:", e);
                settings.propertyAddresses = [];
            }
        }
    try {
        // Generate unique token for tracking
        const token = crypto.randomBytes(32).toString('hex');

        // Create new report entry in database
        const newReport = new Report({
            token,
            status: 'pending',
            userId: req.user.id
        });

        await newReport.save();
        console.log(`[${new Date().toISOString()}] User ${userId}: Created report token ${token}`);

        // Call the actual Flask app
        const FLASK_APP_URL = 'http://127.0.0.1:5000';
        
        // Prepare data for Flask call - match the format expected by Flask
        const state = getStateFromDetails(details);
        let flaskPayload;
        if (zillowlink) {
            flaskPayload = { link: zillowlink, state: state, settings: settings };
        } else {
            // Filter out problematic fields as we do in app.py
            const { Parking, Price, 'Home Type': HomeType, Heating, Cooling, Laundry, Zipcode, ...filteredDetails } = details;
            flaskPayload = { details: filteredDetails, state: state, settings: settings  };
        }

        console.log(`[${new Date().toISOString()}] User ${userId}: Calling Flask app at ${FLASK_APP_URL}/create-report`);
        console.log(`[${new Date().toISOString()}] Flask payload:`, flaskPayload);

        try {
            const flaskResponse = await axios.post(`${FLASK_APP_URL}/create-report`, flaskPayload, {
                timeout: 120000 // 30 second timeout for the ML processing
            });

            const reportData = flaskResponse.data;
            console.log(`[${new Date().toISOString()}] User ${userId}: Flask response received:`, reportData)

            // Complete the report with Flask data
            newReport.status = 'completed';
            newReport.data = reportData;
            await newReport.save();

        } catch (flaskError) {
            console.error(`[${new Date().toISOString()}] User ${userId}: Flask API call failed:`, flaskError.message);

            // Mark report as failed
            newReport.status = 'failed';
            await newReport.save();

            return res.status(500).json({
                error: 'Failed to generate report - machine learning service is unavailable',
                details: flaskError.message
            });
        }

        // Fetch the updated report from database to get final status
        const updatedReport = await Report.findOne({ token });

        const response = {
            token,
            status: updatedReport?.status || 'pending',
            message: updatedReport?.status === 'completed'
                ? 'Report completed successfully!'
                : updatedReport?.status === 'failed'
                    ? 'Report generation failed.'
                    : 'Report generation started. Redirecting to view page.'
        };

        console.log(`[${new Date().toISOString()}] User ${userId}: Returning response:`, response);
        res.json(response);

    } catch (error) {
        console.error(`[${new Date().toISOString()}] User ${userId}: Error creating report:`, error);
        res.status(500).json({ error: 'Failed to initialize report' });
    }
});

// Add endpoint to check report status
app.get('/report-status/:token', auth, async (req, res) => {
    const { token } = req.params;
    const userId = req.user.id;

    console.log(`[${new Date().toISOString()}] User ${userId} calling /report-status for token ${token}`);

    try {
        const report = await Report.findOne({ token, userId: req.user.id });

        if (!report) {
            console.log(`[${new Date().toISOString()}] User ${userId}: Report not found for token ${token}`);
            return res.status(404).json({ error: 'Report not found' });
        }

        const responseData = {
            token: report.token,
            status: report.status,
            ...(report.status === 'completed' ? { data: report.data } : {})
        };

        console.log(`[${new Date().toISOString()}] User ${userId}: Report status ${report.status} for token ${token}`);
        res.json(responseData);

    } catch (error) {
        console.error(`[${new Date().toISOString()}] User ${userId}: Error fetching report status:`, error);
        res.status(500).json({ error: 'Failed to fetch report status' });
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});