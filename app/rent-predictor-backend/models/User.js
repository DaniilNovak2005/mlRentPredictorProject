const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
    username: {
        type: String,
        required: true,
        unique: true,
        trim: true
    },
    email : {
        type: String,
        require: true,
        unique: true,
        trim: true
    },
    password: {
        type: String,
        required: true
    },
    membership: {
        type: String,
        default: "None",
        required: false,
    },
    lastLoggedIn: {
        type: Date,
        default: Date.now(),
        required: false,
    },
    isVerified: {
        type: Boolean,
        default: false
    },
    history: {
        type: Array,
        default: []
    },
    settings: {
        type: Object,
        default: {}
    }
});

module.exports = mongoose.model('User', UserSchema);