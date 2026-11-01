#pragma once

#include <hip_session.hpp>

class HipTestBase : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_session = std::make_unique<HipSession>();
    }
    void TearDown() override { m_session = nullptr; }

    int get_device() const { return m_session->get_device(); }
    bool has_device() const { return m_session->has_device(); }

    HipSession& get_session() {
        assert(m_session);
        return *m_session;
    }

private:
    std::unique_ptr<HipSession> m_session;
};

